//! Explicit upstream/semantic stall detection.
//!
//! Timers are O(1) per chunk. Downstream Anthropic pings must never call
//! [`StreamWatch::on_upstream_bytes`] or [`StreamWatch::on_semantic`].

use std::time::Duration;

use tokio::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct StreamTimeouts {
    pub first_event: Duration,
    pub first_semantic: Duration,
    pub stream_idle: Duration,
    pub semantic_idle: Duration,
}

impl StreamTimeouts {
    pub fn from_secs(
        first_event: u64,
        first_semantic: u64,
        stream_idle: u64,
        semantic_idle: u64,
    ) -> Self {
        Self {
            first_event: Duration::from_secs(first_event),
            first_semantic: Duration::from_secs(first_semantic),
            stream_idle: Duration::from_secs(stream_idle),
            semantic_idle: Duration::from_secs(semantic_idle),
        }
    }

    pub fn disabled() -> Self {
        Self {
            first_event: Duration::ZERO,
            first_semantic: Duration::ZERO,
            stream_idle: Duration::ZERO,
            semantic_idle: Duration::ZERO,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StallKind {
    FirstEvent,
    FirstSemantic,
    StreamIdle,
    SemanticIdle,
}

impl StallKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FirstEvent => "upstream_first_event_timeout",
            Self::FirstSemantic => "upstream_first_semantic_timeout",
            Self::StreamIdle => "upstream_stream_idle_timeout",
            Self::SemanticIdle => "upstream_semantic_idle_timeout",
        }
    }

    pub fn message(self) -> &'static str {
        match self {
            Self::FirstEvent => "upstream produced no stream event before the first-event timeout",
            Self::FirstSemantic => {
                "upstream produced no reasoning/text/tool progress before the first-semantic timeout"
            }
            Self::StreamIdle => "upstream stream was idle (no bytes) beyond the configured limit",
            Self::SemanticIdle => {
                "upstream produced no reasoning/text/tool progress beyond the semantic idle limit"
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamWatch {
    timeouts: StreamTimeouts,
    started: Instant,
    last_byte: Instant,
    last_semantic: Instant,
    got_first_event: bool,
    got_first_semantic: bool,
}

impl StreamWatch {
    pub fn new(timeouts: StreamTimeouts, started: Instant) -> Self {
        Self {
            timeouts,
            started,
            last_byte: started,
            last_semantic: started,
            got_first_event: false,
            got_first_semantic: false,
        }
    }

    pub fn on_upstream_bytes(&mut self, now: Instant) {
        self.last_byte = now;
    }

    pub fn on_sse_event(&mut self) {
        self.got_first_event = true;
    }

    pub fn on_semantic(&mut self, now: Instant) {
        self.got_first_semantic = true;
        self.last_semantic = now;
    }

    pub fn saw_first_event(&self) -> bool {
        self.got_first_event
    }

    pub fn saw_semantic(&self) -> bool {
        self.got_first_semantic
    }

    pub fn check(&self, now: Instant) -> Option<StallKind> {
        if !self.got_first_event
            && !self.timeouts.first_event.is_zero()
            && now.saturating_duration_since(self.started) >= self.timeouts.first_event
        {
            return Some(StallKind::FirstEvent);
        }
        if !self.got_first_semantic
            && !self.timeouts.first_semantic.is_zero()
            && now.saturating_duration_since(self.started) >= self.timeouts.first_semantic
        {
            return Some(StallKind::FirstSemantic);
        }
        if !self.timeouts.stream_idle.is_zero()
            && now.saturating_duration_since(self.last_byte) >= self.timeouts.stream_idle
        {
            return Some(StallKind::StreamIdle);
        }
        if self.got_first_semantic
            && !self.timeouts.semantic_idle.is_zero()
            && now.saturating_duration_since(self.last_semantic) >= self.timeouts.semantic_idle
        {
            return Some(StallKind::SemanticIdle);
        }
        None
    }

    /// Next time a stall may fire. Far-future when every timer is disabled.
    pub fn next_deadline(&self) -> Instant {
        let mut deadline = self.started + Duration::from_secs(60 * 60 * 24 * 365);
        if !self.got_first_event && !self.timeouts.first_event.is_zero() {
            deadline = deadline.min(self.started + self.timeouts.first_event);
        }
        if !self.got_first_semantic && !self.timeouts.first_semantic.is_zero() {
            deadline = deadline.min(self.started + self.timeouts.first_semantic);
        }
        if !self.timeouts.stream_idle.is_zero() {
            deadline = deadline.min(self.last_byte + self.timeouts.stream_idle);
        }
        if self.got_first_semantic && !self.timeouts.semantic_idle.is_zero() {
            deadline = deadline.min(self.last_semantic + self.timeouts.semantic_idle);
        }
        deadline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watch_at(start: Instant) -> StreamWatch {
        StreamWatch::new(StreamTimeouts::from_secs(3, 5, 2, 4), start)
    }

    #[test]
    fn first_event_timeout_before_any_bytes() {
        let start = Instant::from_std(std::time::Instant::now());
        let watch = StreamWatch::new(StreamTimeouts::from_secs(3, 30, 30, 30), start);
        assert_eq!(watch.check(start + Duration::from_secs(2)), None);
        assert_eq!(
            watch.check(start + Duration::from_secs(3)),
            Some(StallKind::FirstEvent)
        );
    }

    #[test]
    fn first_semantic_timeout_despite_nonsemantic_frames() {
        let start = Instant::from_std(std::time::Instant::now());
        let mut watch = watch_at(start);
        let t1 = start + Duration::from_secs(1);
        watch.on_upstream_bytes(t1);
        watch.on_sse_event();
        let t4 = start + Duration::from_secs(4);
        watch.on_upstream_bytes(t4);
        assert_eq!(watch.check(t4), None);
        assert_eq!(
            watch.check(start + Duration::from_secs(5)),
            Some(StallKind::FirstSemantic)
        );
    }

    #[test]
    fn stream_idle_timeout_after_bytes_stop() {
        let start = Instant::from_std(std::time::Instant::now());
        let mut watch = watch_at(start);
        watch.on_upstream_bytes(start);
        watch.on_sse_event();
        watch.on_semantic(start);
        assert_eq!(
            watch.check(start + Duration::from_secs(2)),
            Some(StallKind::StreamIdle)
        );
    }

    #[test]
    fn semantic_idle_despite_heartbeat_bytes() {
        let start = Instant::from_std(std::time::Instant::now());
        let mut watch = watch_at(start);
        watch.on_upstream_bytes(start);
        watch.on_sse_event();
        watch.on_semantic(start);
        let later = start + Duration::from_secs(3);
        watch.on_upstream_bytes(later);
        assert_eq!(
            watch.check(start + Duration::from_secs(4)),
            Some(StallKind::SemanticIdle)
        );
    }

    #[test]
    fn semantic_timer_resets_on_reasoning_text_or_tool_delta() {
        let start = Instant::from_std(std::time::Instant::now());
        let mut watch = watch_at(start);
        watch.on_upstream_bytes(start);
        watch.on_sse_event();
        watch.on_semantic(start);
        let t = start + Duration::from_secs(3);
        watch.on_upstream_bytes(t);
        watch.on_semantic(t);
        let later = t + Duration::from_secs(3);
        watch.on_upstream_bytes(later);
        assert_eq!(watch.check(later), None);
        assert_eq!(
            watch.check(t + Duration::from_secs(4)),
            Some(StallKind::SemanticIdle)
        );
    }

    #[test]
    fn local_ping_does_not_reset_upstream_timers() {
        let start = Instant::from_std(std::time::Instant::now());
        let mut watch = watch_at(start);
        watch.on_upstream_bytes(start);
        watch.on_sse_event();
        // Simulate a local ping: no on_upstream_bytes / on_semantic.
        assert_eq!(
            watch.check(start + Duration::from_secs(2)),
            Some(StallKind::StreamIdle)
        );
    }

    #[test]
    fn zero_duration_disables_that_timer() {
        let start = Instant::from_std(std::time::Instant::now());
        let watch = StreamWatch::new(StreamTimeouts::disabled(), start);
        assert_eq!(watch.check(start + Duration::from_secs(10_000)), None);
    }

    #[test]
    fn stall_kind_names_are_stable() {
        assert_eq!(
            StallKind::FirstEvent.as_str(),
            "upstream_first_event_timeout"
        );
        assert_eq!(
            StallKind::FirstSemantic.as_str(),
            "upstream_first_semantic_timeout"
        );
        assert_eq!(
            StallKind::StreamIdle.as_str(),
            "upstream_stream_idle_timeout"
        );
        assert_eq!(
            StallKind::SemanticIdle.as_str(),
            "upstream_semantic_idle_timeout"
        );
    }
}
