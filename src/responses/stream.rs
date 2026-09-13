//! OpenAI Chat Completions SSE → Responses API SSE conversion.
//!
//! Event lifecycle follows the wire schema consumed by Grok Build (typed
//! `type`-tagged frames; a frame with an unknown `type` or a missing required
//! field fails the client's deserialization, so every field here is exact):
//!
//! ```text
//! response.created
//! response.in_progress
//! response.output_item.added        (reasoning)
//! response.reasoning_summary_part.added
//! response.reasoning_summary_text.delta …
//! response.reasoning_summary_text.done
//! response.reasoning_summary_part.done
//! response.output_item.added        (message)
//! response.content_part.added
//! response.output_text.delta …
//! response.output_text.done
//! response.content_part.done
//! response.output_item.done
//! response.output_item.added        (function_call)   ← ALWAYS before args
//! response.function_call_arguments.delta …
//! response.function_call_arguments.done
//! response.output_item.done
//! response.completed | response.incomplete | response.failed
//! ```
//!
//! Grok Build maps `output_index` → its internal tool index ONLY when the
//! FunctionCall arrives in `response.output_item.added`; argument deltas for
//! an unmapped index are silently dropped. This converter therefore never
//! emits argument deltas before the item-added frame, and never emits an
//! item-added without `call_id`/`name`.
//!
//! The same machinery accumulates state for the terminal response object
//! (Grok Build replays `response.completed.response.output` as next-turn
//! conversation state) and for non-stream aggregation — streaming and
//! non-streaming cannot diverge.

use serde_json::{json, Value};

use super::types::{
    map_usage, new_function_call_id, new_message_id, new_reasoning_id, new_response_id, sse_event,
};

/// One upstream tool call being accumulated (indexed by the upstream
/// `tool_calls[].index`).
#[derive(Debug)]
struct ToolSlot {
    item_id: String,
    call_id: String,
    name: String,
    arguments: String,
    /// Responses `output_index` assigned when the item was added.
    output_index: usize,
    done: bool,
    /// A server-tool call (`web_search`): accumulated for the search loop
    /// but never emitted to the client as a `function_call` item — the
    /// proxy executes it and answers with `web_search_call` items (see
    /// [`ResponsesConverter::emit_web_search_round`]).
    server: bool,
}

/// One output item in allocation order (terminal reconstruction ordering).
/// Text lives in the live accumulators; these descriptors pin the order and
/// identity only. `WebSearchCall` carries its full item (the proxy authors
/// it — the upstream never emits it).
#[derive(Debug)]
enum OutputItem {
    Reasoning { item_id: String },
    Message { item_id: String },
    FunctionCall { item_id: String },
    WebSearchCall { item: Value },
}

/// The streaming converter. Feed parsed OpenAI chat chunks; emitted Responses
/// SSE frames accumulate in the caller's buffer.
pub struct ResponsesConverter {
    response_id: String,
    model: String,
    /// Whether reasoning may be surfaced (requested via `reasoning.summary`).
    expose_thinking: bool,
    started: bool,
    sequence: u64,
    created_at: u64,

    /// Next Responses `output_index` (per item, in emission order).
    next_output_index: usize,

    reasoning: Option<ReasoningState>,
    message: Option<MessageState>,

    tools: std::collections::BTreeMap<i64, ToolSlot>,
    /// Function names executed server-side (see `crate::websearch`).
    server_tool_names: std::collections::HashSet<String>,
    /// Completed output items in allocation order (terminal reconstruction).
    output_items: Vec<OutputItem>,

    finish_reason: Option<String>,
    /// (input, output, cached, reasoning) tokens.
    usage: Option<(u64, u64, u64, u64)>,
    /// Set when the upstream stream failed abnormally.
    error: Option<String>,
    model_seen: Option<String>,
}

#[derive(Debug)]
struct ReasoningState {
    item_id: String,
    output_index: usize,
    accum: String,
    item_added: bool,
    part_added: bool,
    part_done: bool,
    item_done: bool,
}

#[derive(Debug)]
struct MessageState {
    item_id: String,
    output_index: usize,
    accum: String,
    item_added: bool,
    part_added: bool,
    part_done: bool,
    item_done: bool,
}

impl ResponsesConverter {
    pub fn new(model: &str, expose_thinking: bool) -> Self {
        Self::with_server_tools(model, expose_thinking, std::collections::HashSet::new())
    }

    /// Converter that suppresses client-side `function_call` items for the
    /// given upstream function names (server-executed tools).
    pub fn with_server_tools(
        model: &str,
        expose_thinking: bool,
        server_tool_names: std::collections::HashSet<String>,
    ) -> Self {
        Self {
            response_id: new_response_id(),
            model: model.to_owned(),
            expose_thinking,
            started: false,
            sequence: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            next_output_index: 0,
            reasoning: None,
            message: None,
            tools: std::collections::BTreeMap::new(),
            server_tool_names,
            output_items: Vec::new(),
            finish_reason: None,
            usage: None,
            error: None,
            model_seen: None,
        }
    }

    pub fn model(&self) -> &str {
        self.model_seen.as_deref().unwrap_or(&self.model)
    }

    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// Cheap progress probes for the pump (stats + watchdog bookkeeping).
    pub fn has_reasoning(&self) -> bool {
        self.reasoning.as_ref().is_some_and(|r| !r.accum.is_empty())
    }

    pub fn has_text(&self) -> bool {
        self.message.as_ref().is_some_and(|m| !m.accum.is_empty())
    }

    pub fn has_tools(&self) -> bool {
        !self.tools.is_empty()
    }

    /// Exposed reasoning text (shadow bookkeeping uses the same in-turn
    /// continuity rule as the Anthropic frontend).
    pub fn reasoning_text(&self) -> &str {
        self.reasoning
            .as_ref()
            .map(|r| r.accum.as_str())
            .unwrap_or("")
    }

    pub fn tool_call_ids(&self) -> Vec<String> {
        self.tools
            .values()
            .map(|slot| slot.call_id.clone())
            .collect()
    }

    /// Accumulated byte sizes (summary logging; never content).
    pub fn reasoning_bytes(&self) -> u64 {
        self.reasoning.as_ref().map_or(0, |r| r.accum.len() as u64)
    }

    pub fn text_bytes(&self) -> u64 {
        self.message.as_ref().map_or(0, |m| m.accum.len() as u64)
    }

    pub fn tool_bytes(&self) -> u64 {
        self.tools
            .values()
            .map(|slot| slot.arguments.len() as u64)
            .sum()
    }

    pub fn usage(&self) -> Option<(u64, u64, u64, u64)> {
        self.usage
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence += 1;
        self.sequence
    }

    fn emit(&mut self, mut event: Value) -> String {
        if let Some(object) = event.as_object_mut() {
            object.insert("sequence_number".into(), json!(self.next_sequence()));
        }
        sse_event(&event)
    }

    /// Emit `response.created` + `response.in_progress` on the first upstream
    /// chunk (streaming starts immediately; nothing is buffered waiting for
    /// semantic content).
    fn ensure_started(&mut self, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        let response = super::types::base_response(
            &self.response_id,
            self.model(),
            "in_progress",
            Vec::new(),
            None,
            self.created_at,
        );
        let created = self.emit(json!({"type": "response.created", "response": response.clone()}));
        out.extend_from_slice(created.as_bytes());
        let in_progress = self.emit(json!({"type": "response.in_progress", "response": response}));
        out.extend_from_slice(in_progress.as_bytes());
    }

    /// Feed one parsed OpenAI chat chunk; append emitted Responses frames to
    /// `out`. Returns true when the chunk carried semantic progress.
    pub fn feed_chunk(&mut self, chunk: &Value, out: &mut Vec<u8>) -> bool {
        self.ensure_started(out);
        let mut semantic = false;
        if let Some(model) = chunk.get("model").and_then(Value::as_str) {
            if !model.is_empty() {
                self.model_seen = Some(model.to_owned());
            }
        }
        if let Some(usage) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(map_usage(usage));
        }
        let Some(choices) = chunk.get("choices").and_then(Value::as_array) else {
            return false;
        };
        for choice in choices {
            if let Some(finish) = choice.get("finish_reason").and_then(Value::as_str) {
                self.finish_reason = Some(finish.to_owned());
                semantic = true;
            }
            let Some(delta) = choice.get("delta") else {
                continue;
            };
            // reasoning_content → reasoning item (only when requested).
            if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                if !reasoning.is_empty() {
                    semantic = true;
                    self.feed_reasoning(reasoning, out);
                }
            }
            // content → message text.
            if let Some(content) = delta.get("content").and_then(Value::as_str) {
                if !content.is_empty() {
                    semantic = true;
                    self.feed_text(content, out);
                }
            }
            // tool_calls → function_call items (parallel-capable). Two
            // passes: item-added frames for every new index FIRST, then
            // argument deltas — a delta must never precede its item.
            if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                semantic = true;
                for tool_call in tool_calls {
                    self.ensure_tool_item_added(tool_call, out);
                }
                for tool_call in tool_calls {
                    self.feed_tool_call_arguments(tool_call, out);
                }
            }
        }
        semantic
    }

    fn feed_reasoning(&mut self, reasoning: &str, out: &mut Vec<u8>) {
        if !self.expose_thinking {
            // Not requested: never fabricate or expose reasoning (the
            // accumulator still records it for in-turn shadow continuity).
            if let Some(state) = self.reasoning.as_mut() {
                state.accum.push_str(reasoning);
            } else {
                self.reasoning = Some(ReasoningState {
                    item_id: new_reasoning_id(),
                    output_index: usize::MAX,
                    accum: reasoning.to_owned(),
                    item_added: false,
                    part_added: false,
                    part_done: false,
                    item_done: false,
                });
            }
            return;
        }
        if self.reasoning.is_none() {
            self.reasoning = Some(ReasoningState {
                item_id: new_reasoning_id(),
                output_index: usize::MAX,
                accum: String::new(),
                item_added: false,
                part_added: false,
                part_done: false,
                item_done: false,
            });
        }
        let (item_id, output_index, item_first, part_first) = {
            let state = self.reasoning.as_mut().expect("reasoning state");
            state.accum.push_str(reasoning);
            let item_first = !state.item_added;
            state.item_added = true;
            if state.output_index == usize::MAX {
                state.output_index = self.next_output_index;
                self.next_output_index += 1;
            }
            let part_first = !state.part_added;
            state.part_added = true;
            (
                state.item_id.clone(),
                state.output_index,
                item_first,
                part_first,
            )
        };
        if item_first {
            self.output_items.push(OutputItem::Reasoning {
                item_id: item_id.clone(),
            });
            let event = self.emit(json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [],
                    "status": "in_progress"
                }
            }));
            out.extend_from_slice(event.as_bytes());
        }
        if part_first {
            let event = self.emit(json!({
                "type": "response.reasoning_summary_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}
            }));
            out.extend_from_slice(event.as_bytes());
        }
        let event = self.emit(json!({
            "type": "response.reasoning_summary_text.delta",
            "item_id": item_id,
            "output_index": output_index,
            "summary_index": 0,
            "delta": reasoning
        }));
        out.extend_from_slice(event.as_bytes());
    }

    fn feed_text(&mut self, content: &str, out: &mut Vec<u8>) {
        if self.message.is_none() {
            self.message = Some(MessageState {
                item_id: new_message_id(),
                output_index: usize::MAX,
                accum: String::new(),
                item_added: false,
                part_added: false,
                part_done: false,
                item_done: false,
            });
        }
        let (item_id, output_index, first) = {
            let state = self.message.as_mut().expect("message state");
            state.accum.push_str(content);
            let first = !state.item_added;
            if first {
                state.item_added = true;
                state.part_added = true;
            }
            if state.output_index == usize::MAX {
                state.output_index = self.next_output_index;
                self.next_output_index += 1;
            }
            (state.item_id.clone(), state.output_index, first)
        };
        if first {
            self.output_items.push(OutputItem::Message {
                item_id: item_id.clone(),
            });
            let event = self.emit(json!({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": {
                    "type": "message",
                    "id": item_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }));
            out.extend_from_slice(event.as_bytes());
            let event = self.emit(json!({
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}
            }));
            out.extend_from_slice(event.as_bytes());
        }
        let event = self.emit(json!({
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "delta": content
        }));
        out.extend_from_slice(event.as_bytes());
    }

    /// Pass 1: register the tool slot and emit its `output_item.added` frame
    /// (call_id + name included) if this upstream index is new.
    fn ensure_tool_item_added(&mut self, tool_call: &Value, out: &mut Vec<u8>) {
        let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
        if self.tools.contains_key(&index) {
            return;
        }
        // Close the message item before the first function-call item: output
        // items are sequential; a message still open when a tool item starts
        // would emit an inconsistent ordering to the client.
        if self.tools.is_empty() {
            self.close_message_item(out);
        }
        let call_id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let name = tool_call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        let is_server = self.server_tool_names.contains(&name);
        let output_index = self.next_output_index;
        self.next_output_index += 1;
        let item_id = new_function_call_id();
        self.tools.insert(
            index,
            ToolSlot {
                item_id: item_id.clone(),
                call_id: call_id.clone(),
                name: name.clone(),
                arguments: String::new(),
                output_index,
                done: false,
                server: is_server,
            },
        );
        // Server-tool calls never reach the client as `function_call` items:
        // the proxy executes them and emits `web_search_call` items instead.
        if is_server {
            return;
        }
        // Item-added FIRST: Grok Build builds its output_index → tool mapping
        // (and needs call_id + name) from this frame alone. An argument delta
        // emitted before it would be silently dropped.
        let event = self.emit(json!({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": {
                "type": "function_call",
                "id": item_id,
                "call_id": call_id,
                "name": name,
                "arguments": "",
                "status": "in_progress"
            }
        }));
        out.extend_from_slice(event.as_bytes());
        self.output_items.push(OutputItem::FunctionCall { item_id });
    }

    /// Pass 2: accumulate + emit argument deltas for an existing tool slot.
    fn feed_tool_call_arguments(&mut self, tool_call: &Value, out: &mut Vec<u8>) {
        let index = tool_call.get("index").and_then(Value::as_i64).unwrap_or(0);
        let Some(slot) = self.tools.get_mut(&index) else {
            return;
        };
        // Fragmented name / late id: upstream sends the name with the first
        // delta in practice; patch when it arrives fragmented.
        if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
            if !id.is_empty() {
                slot.call_id = id.to_owned();
            }
        }
        if let Some(name) = tool_call.pointer("/function/name").and_then(Value::as_str) {
            if !name.is_empty() {
                slot.name = name.to_owned();
            }
        }
        let mut delta_args = String::new();
        if let Some(arguments) = tool_call
            .pointer("/function/arguments")
            .and_then(Value::as_str)
        {
            slot.arguments.push_str(arguments);
            delta_args.push_str(arguments);
        }
        let (item_id, output_index) = (slot.item_id.clone(), slot.output_index);
        let is_server = slot.server;
        if !delta_args.is_empty() {
            // Argument deltas for server tools stay internal (accumulated in
            // the slot above); the client sees a `web_search_call` item.
            if !is_server {
                let event = self.emit(json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item_id,
                    "output_index": output_index,
                    "delta": delta_args
                }));
                out.extend_from_slice(event.as_bytes());
            }
        }
    }

    /// Close the open message item (part done → text done → item done).
    /// Idempotent; called before tool items start and at finish.
    fn close_message_item(&mut self, out: &mut Vec<u8>) {
        let Some(state) = &self.message else {
            return;
        };
        if state.item_done {
            return;
        }
        let (item_id, output_index, text, part_open) = (
            state.item_id.clone(),
            state.output_index,
            state.accum.clone(),
            state.part_added && !state.part_done,
        );
        let state = self.message.as_mut().expect("message state");
        state.item_done = true;
        state.part_done = true;
        if part_open {
            let event = self.emit(json!({
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text
            }));
            out.extend_from_slice(event.as_bytes());
            let event = self.emit(json!({
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []}
            }));
            out.extend_from_slice(event.as_bytes());
        }
        let event = self.emit(json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "message",
                "id": item_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]
            }
        }));
        out.extend_from_slice(event.as_bytes());
    }

    /// Close the open reasoning part/item. Idempotent.
    fn close_reasoning_item(&mut self, out: &mut Vec<u8>) {
        let Some(state) = &self.reasoning else {
            return;
        };
        if !state.item_added || state.item_done {
            return;
        }
        let (item_id, output_index, text, part_open) = (
            state.item_id.clone(),
            state.output_index,
            state.accum.clone(),
            state.part_added && !state.part_done,
        );
        let state = self.reasoning.as_mut().expect("reasoning state");
        state.item_done = true;
        state.part_done = true;
        if part_open {
            let event = self.emit(json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "text": text
            }));
            out.extend_from_slice(event.as_bytes());
            let event = self.emit(json!({
                "type": "response.reasoning_summary_part.done",
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": text}
            }));
            out.extend_from_slice(event.as_bytes());
        }
        let event = self.emit(json!({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": {
                "type": "reasoning",
                "id": item_id,
                "summary": [{"type": "summary_text", "text": text}],
                "status": "completed"
            }
        }));
        out.extend_from_slice(event.as_bytes());
    }

    /// Complete the logical stream: close open items, emit `done` frames for
    /// every tool call, then the terminal event. `error` describes an
    /// abnormal termination (EOF without DONE, transport error, stall) —
    /// never a retry (semantic output may already have streamed).
    pub fn finish(&mut self, error: Option<&str>, out: &mut Vec<u8>) {
        self.error = error.map(str::to_owned);
        // A terminal error before any semantic content: surface
        // `response.failed` (Grok Build converts it into a typed sampling
        // error; the response object carries the sanitized message).
        if let Some(message) = &self.error {
            let response = self.terminal_response("failed");
            let mut object = response;
            object["error"] = json!({"code": "upstream_error", "message": message});
            let event = self.emit(json!({"type": "response.failed", "response": object}));
            out.extend_from_slice(event.as_bytes());
            return;
        }
        self.close_reasoning_item(out);
        self.close_message_item(out);
        // Argument `done` + item `done` for every client tool call, in
        // output order. Server-tool slots are skipped (never client-visible
        // as function_call items).
        let slots: Vec<(String, usize, String, String, String)> = self
            .tools
            .values()
            .filter(|slot| !slot.done && !slot.server)
            .map(|slot| {
                (
                    slot.item_id.clone(),
                    slot.output_index,
                    slot.call_id.clone(),
                    slot.name.clone(),
                    slot.arguments.clone(),
                )
            })
            .collect();
        for (item_id, output_index, call_id, name, arguments) in slots {
            let event = self.emit(json!({
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": output_index,
                "arguments": arguments
            }));
            out.extend_from_slice(event.as_bytes());
            let event = self.emit(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": {
                    "type": "function_call",
                    "id": item_id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed"
                }
            }));
            out.extend_from_slice(event.as_bytes());
        }
        for slot in self.tools.values_mut() {
            slot.done = true;
        }
        // Terminal frame: completed, or incomplete (max_output_tokens) when
        // the upstream hit the length cap. Grok Build requires one of these;
        // deltas + [DONE] alone are not a valid Responses stream.
        let (status, incomplete_details) = match self.finish_reason.as_deref() {
            Some("length") => ("incomplete", Some("max_output_tokens")),
            _ => ("completed", None),
        };
        let mut response = self.terminal_response(status);
        if let Some(reason) = incomplete_details {
            response["incomplete_details"] = json!({"reason": reason});
        }
        let event_type = if status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        let event = self.emit(json!({"type": event_type, "response": response}));
        out.extend_from_slice(event.as_bytes());
    }

    /// True when any upstream function call in this response is server-side.
    pub fn has_server_tools(&self) -> bool {
        self.tools.values().any(|slot| slot.server)
    }

    /// True when any upstream function call is a client tool call (a mixed
    /// round: server tools execute, then the response ends so the client can
    /// run its own tools).
    pub fn has_client_tools(&self) -> bool {
        self.tools.values().any(|slot| !slot.server)
    }

    /// Collected server-tool calls (query included).
    pub fn server_tool_calls(&self) -> Vec<crate::websearch::OpenAiToolCall> {
        self.tools
            .values()
            .filter(|slot| slot.server)
            .map(|slot| crate::websearch::OpenAiToolCall {
                id: slot.call_id.clone(),
                name: slot.name.clone(),
                arguments: slot.arguments.clone(),
            })
            .collect()
    }

    /// Round boundary after a server-tool continuation: the upstream call
    /// accumulator is per-round (the next upstream generation starts fresh);
    /// client-facing accumulators (text, reasoning) and the recorded
    /// `web_search_call` output items persist for the terminal response.
    pub fn begin_new_upstream_round(&mut self) {
        self.tools.clear();
        self.finish_reason = None;
    }

    /// Emit one gateway-executed web-search round as Responses SSE:
    /// `response.web_search_call.in_progress` per call, then — once the
    /// executions settle — `response.output_item.done` carrying the full
    /// `web_search_call` item (query + source URLs + status). Returns the
    /// number of executed searches. Each item is also recorded for terminal
    /// reconstruction (Grok Build replays `response.completed.output` as
    /// next-turn conversation state).
    pub fn emit_web_search_round(
        &mut self,
        calls: &[crate::websearch::OpenAiToolCall],
        outcomes: &[crate::websearch::SearchOutcome],
        out: &mut Vec<u8>,
    ) -> u64 {
        for (_call, outcome) in calls.iter().zip(outcomes.iter()) {
            let item_id = format!("ws_{}", uuid::Uuid::new_v4().simple());
            let output_index = self.next_output_index;
            self.next_output_index += 1;
            let started = self.emit(json!({
                "type": "response.web_search_call.in_progress",
                "output_index": output_index,
                "item_id": item_id
            }));
            out.extend_from_slice(started.as_bytes());
            let sources: Vec<Value> = outcome
                .results
                .iter()
                .filter_map(|result| {
                    result
                        .get("url")
                        .and_then(Value::as_str)
                        .map(|url| json!({"type": "url", "url": url}))
                })
                .collect();
            let item = json!({
                "type": "web_search_call",
                "id": item_id,
                "status": if outcome.error.is_some() { "failed" } else { "completed" },
                "action": {
                    "type": "search",
                    "query": outcome.query,
                    "sources": sources
                }
            });
            let done = self.emit(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
            }));
            out.extend_from_slice(done.as_bytes());
            self.output_items.push(OutputItem::WebSearchCall { item });
        }
        outcomes.len() as u64
    }

    /// Reconstruct the full response object for the terminal event (and
    /// non-stream aggregation) from the accumulated output items.
    fn terminal_response(&self, status: &str) -> Value {
        let mut output = Vec::with_capacity(self.output_items.len());
        for item in &self.output_items {
            match item {
                OutputItem::Reasoning { item_id } => {
                    let text = self
                        .reasoning
                        .as_ref()
                        .filter(|r| &r.item_id == item_id)
                        .map(|r| r.accum.as_str())
                        .unwrap_or("");
                    output.push(json!({
                        "type": "reasoning",
                        "id": item_id,
                        "summary": [{"type": "summary_text", "text": text}],
                        "status": "completed"
                    }));
                }
                OutputItem::Message { item_id } => {
                    let text = self
                        .message
                        .as_ref()
                        .filter(|m| &m.item_id == item_id)
                        .map(|m| m.accum.as_str())
                        .unwrap_or("");
                    output.push(json!({
                        "type": "message",
                        "id": item_id,
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}]
                    }));
                }
                OutputItem::FunctionCall { item_id } => {
                    let Some(slot) = self.tools.values().find(|slot| &slot.item_id == item_id)
                    else {
                        continue;
                    };
                    output.push(json!({
                        "type": "function_call",
                        "id": slot.item_id,
                        "call_id": slot.call_id,
                        "name": slot.name,
                        "arguments": slot.arguments,
                        "status": "completed"
                    }));
                }
                OutputItem::WebSearchCall { item } => {
                    output.push(item.clone());
                }
            }
        }
        super::types::base_response(
            &self.response_id,
            self.model(),
            status,
            output,
            self.usage,
            self.created_at,
        )
    }

    /// Assemble the full non-stream Responses object from the same
    /// accumulated state the streaming path uses (cannot diverge).
    pub fn nonstream_response(&self) -> Value {
        let status = match (self.error.as_deref(), self.finish_reason.as_deref()) {
            (Some(_), _) => "failed",
            (None, Some("length")) => "incomplete",
            _ => "completed",
        };
        let mut response = self.terminal_response(status);
        if status == "failed" {
            response["error"] = json!({
                "code": "upstream_error",
                "message": self.error.as_deref().unwrap_or("upstream error")
            });
        }
        if status == "incomplete" {
            response["incomplete_details"] = json!({"reason": "max_output_tokens"});
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(delta: Value, finish: Option<&str>, usage: Option<Value>) -> Value {
        let mut chunk = json!({
            "id": "c1",
            "model": "z-ai/glm-5.3-flash",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
        });
        if let Some(usage) = usage {
            chunk["usage"] = usage;
        }
        chunk
    }

    /// Parse emitted Responses SSE frames into `(type, data)` pairs.
    fn parse_frames(raw: &[u8]) -> Vec<(String, Value)> {
        let text = String::from_utf8_lossy(raw);
        let mut events = Vec::new();
        for block in text.split("\n\n") {
            for line in block.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    let value: Value = serde_json::from_str(data).unwrap();
                    events.push((value["type"].as_str().unwrap().to_owned(), value));
                }
            }
        }
        events
    }

    #[test]
    fn text_stream_lifecycle_and_terminal_reconstruction() {
        let mut converter = ResponsesConverter::new("z-ai/glm-5.3-flash", false);
        let mut out = Vec::new();
        converter.feed_chunk(&chunk(json!({"content": "Hel"}), None, None), &mut out);
        converter.feed_chunk(&chunk(json!({"content": "lo"}), None, None), &mut out);
        converter.feed_chunk(&chunk(json!({}), Some("stop"), None), &mut out);
        converter.finish(None, &mut out);
        let frames = parse_frames(&out);
        let types: Vec<&str> = frames.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        // Monotonically increasing sequence numbers.
        let sequences: Vec<u64> = frames
            .iter()
            .map(|(_, v)| v["sequence_number"].as_u64().unwrap())
            .collect();
        assert!(sequences.windows(2).all(|w| w[1] == w[0] + 1));
        // Terminal output reconstructs the message with full text.
        let terminal = &frames.last().unwrap().1;
        assert_eq!(terminal["response"]["status"], "completed");
        assert_eq!(terminal["response"]["output"][0]["type"], "message");
        assert_eq!(
            terminal["response"]["output"][0]["content"][0]["text"],
            "Hello"
        );
        assert_eq!(terminal["response"]["usage"]["total_tokens"], 0);
    }

    #[test]
    fn tool_calls_item_added_precedes_argument_deltas() {
        let mut converter = ResponsesConverter::new("m", false);
        let mut out = Vec::new();
        converter.feed_chunk(
            &chunk(
                json!({"tool_calls": [
                    {"index": 0, "id": "call_a", "type": "function",
                     "function": {"name": "Read", "arguments": "{\"pa"}},
                    {"index": 1, "id": "call_b", "type": "function",
                     "function": {"name": "Edit", "arguments": "{\"pa"}}
                ]}),
                None,
                None,
            ),
            &mut out,
        );
        converter.feed_chunk(
            &chunk(
                json!({"tool_calls": [
                    {"index": 0, "function": {"arguments": "th\": \"a.rs\"}"}},
                    {"index": 1, "function": {"arguments": "th\": \"b.rs\"}"}}
                ]}),
                None,
                None,
            ),
            &mut out,
        );
        converter.feed_chunk(&chunk(json!({}), Some("tool_calls"), None), &mut out);
        converter.finish(None, &mut out);
        let frames = parse_frames(&out);
        let types: Vec<&str> = frames.iter().map(|(t, _)| t.as_str()).collect();
        // item.added frames (both tools) come before ANY arguments delta.
        let first_delta = types
            .iter()
            .position(|t| *t == "response.function_call_arguments.delta")
            .unwrap();
        let added_count = types[..first_delta]
            .iter()
            .filter(|t| **t == "response.output_item.added")
            .count();
        assert_eq!(
            added_count, 2,
            "both function_call items must be added before the first args delta: {types:?}"
        );
        // call_id/name on the added items.
        let added: Vec<&Value> = frames
            .iter()
            .filter(|(t, _)| t == "response.output_item.added")
            .map(|(_, v)| &v["item"])
            .collect();
        assert_eq!(added[0]["call_id"], "call_a");
        assert_eq!(added[0]["name"], "Read");
        assert_eq!(added[1]["call_id"], "call_b");
        // Deltas route to the right output_index and reconstruct fully.
        let mut by_index: std::collections::BTreeMap<u64, String> = Default::default();
        for (t, v) in &frames {
            if t == "response.function_call_arguments.delta" {
                by_index
                    .entry(v["output_index"].as_u64().unwrap())
                    .or_default()
                    .push_str(v["delta"].as_str().unwrap());
            }
        }
        assert_eq!(by_index[&0], "{\"path\": \"a.rs\"}");
        assert_eq!(by_index[&1], "{\"path\": \"b.rs\"}");
        // Terminal output has both completed calls with identical call_ids.
        let terminal = &frames.last().unwrap().1;
        let output = terminal["response"]["output"].as_array().unwrap();
        assert_eq!(output[0]["call_id"], "call_a");
        assert_eq!(output[0]["arguments"], "{\"path\": \"a.rs\"}");
        assert_eq!(output[1]["call_id"], "call_b");
        assert_eq!(output[1]["status"], "completed");
        assert_eq!(terminal["response"]["status"], "completed");
    }

    #[test]
    fn reasoning_exposed_only_when_requested() {
        let chunks = [
            chunk(json!({"reasoning_content": "think A"}), None, None),
            chunk(json!({"reasoning_content": "think B"}), None, None),
            chunk(json!({"content": "answer"}), None, None),
            chunk(json!({}), Some("stop"), None),
        ];
        // Requested: reasoning item streams before the message item.
        let mut converter = ResponsesConverter::new("m", true);
        let mut out = Vec::new();
        for c in &chunks {
            converter.feed_chunk(c, &mut out);
        }
        converter.finish(None, &mut out);
        let frames = parse_frames(&out);
        let added_types: Vec<String> = frames
            .iter()
            .filter(|(t, _)| t == "response.output_item.added")
            .map(|(_, v)| v["item"]["type"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(added_types, vec!["reasoning", "message"]);
        let deltas: String = frames
            .iter()
            .filter(|(t, _)| t == "response.reasoning_summary_text.delta")
            .map(|(_, v)| v["delta"].as_str().unwrap())
            .collect();
        assert_eq!(deltas, "think Athink B");
        let terminal = &frames.last().unwrap().1;
        assert_eq!(terminal["response"]["output"][0]["type"], "reasoning");
        assert_eq!(
            terminal["response"]["output"][0]["summary"][0]["text"],
            "think Athink B"
        );

        // Not requested: no reasoning frames at all.
        let mut converter = ResponsesConverter::new("m", false);
        let mut out = Vec::new();
        for c in &chunks {
            converter.feed_chunk(c, &mut out);
        }
        converter.finish(None, &mut out);
        let rendered = String::from_utf8_lossy(&out);
        assert!(!rendered.contains("response.reasoning_summary_text.delta"));
        assert!(!rendered.contains("think A"));
    }

    #[test]
    fn length_maps_to_incomplete_with_reason() {
        let mut converter = ResponsesConverter::new("m", false);
        let mut out = Vec::new();
        converter.feed_chunk(
            &chunk(json!({"content": "abc"}), Some("length"), None),
            &mut out,
        );
        converter.finish(None, &mut out);
        let frames = parse_frames(&out);
        let terminal = &frames.last().unwrap().1;
        assert_eq!(terminal["type"], "response.incomplete");
        assert_eq!(terminal["response"]["status"], "incomplete");
        assert_eq!(
            terminal["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }

    #[test]
    fn usage_maps_full_input_with_cached_subset() {
        let mut converter = ResponsesConverter::new("m", false);
        let mut out = Vec::new();
        converter.feed_chunk(
            &chunk(
                json!({"content": "hi"}),
                None,
                Some(json!({
                    "prompt_tokens": 100, "completion_tokens": 10,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 3}
                })),
            ),
            &mut out,
        );
        converter.feed_chunk(&chunk(json!({}), Some("stop"), None), &mut out);
        converter.finish(None, &mut out);
        let frames = parse_frames(&out);
        let usage = &frames.last().unwrap().1["response"]["usage"];
        assert_eq!(usage["input_tokens"], 100);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 40);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 3);
        assert_eq!(usage["total_tokens"], 110);
    }

    #[test]
    fn midstream_failure_emits_response_failed() {
        let mut converter = ResponsesConverter::new("m", false);
        let mut out = Vec::new();
        converter.feed_chunk(&chunk(json!({"content": "partial"}), None, None), &mut out);
        converter.finish(Some("upstream stream ended unexpectedly"), &mut out);
        let frames = parse_frames(&out);
        let terminal = &frames.last().unwrap().1;
        assert_eq!(terminal["type"], "response.failed");
        assert_eq!(terminal["response"]["status"], "failed");
        assert!(terminal["response"]["error"]["message"].as_str().is_some());
    }
}
