//! In-process GLM-5.3-Flash tokenizer (official `tokenizer.json`).
//!
//! The 20 MB tokenizer JSON is embedded gzip-compressed (~3 MB) and parsed
//! once on first use. Loading never touches the network or the filesystem.

use std::sync::OnceLock;

use crate::glm53::CountError;
use tokenizers::Tokenizer;

/// Official zai-org/GLM-5.3-Flash `tokenizer.json`, gzip-compressed.
/// Revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (MIT); see
/// docs/GLM53_FLASH.md.
static TOKENIZER_JSON_GZ: &[u8] =
    include_bytes!("../../tools/glm_reference/assets/tokenizer.json.gz");

static TOKENIZER: OnceLock<Result<Tokenizer, String>> = OnceLock::new();

pub fn tokenizer() -> Result<&'static Tokenizer, CountError> {
    match TOKENIZER.get_or_init(|| {
        let mut json_bytes = Vec::with_capacity(21 * 1024 * 1024);
        let mut reader = flate2::read::GzDecoder::new(TOKENIZER_JSON_GZ);
        if let Err(error) = std::io::Read::read_to_end(&mut reader, &mut json_bytes) {
            return Err(error.to_string());
        }
        Tokenizer::from_bytes(&json_bytes).map_err(|error| error.to_string())
    }) {
        Ok(tokenizer) => Ok(tokenizer),
        // This arm is unreachable unless the embedded asset is corrupted;
        // fail closed without panicking.
        Err(reason) => Err(CountError {
            error_type: "api_error",
            message: format!("embedded GLM tokenizer failed to load: {reason}"),
        }),
    }
}

/// Tokenize a fully rendered prompt. The template includes its own special
/// tokens (`[gMASK]<sop>`, role markers), so no additional special tokens
/// are added.
pub fn encode_no_special(rendered: &str) -> Result<Vec<u32>, CountError> {
    let encoding = tokenizer()?
        .encode(rendered, false)
        .map_err(|error| CountError {
            error_type: "api_error",
            message: format!("tokenization failed: {error}"),
        })?;
    Ok(encoding.get_ids().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_loads_and_matches_official_special_ids() {
        let tokenizer = tokenizer().unwrap();
        // Added-token IDs from the official tokenizer.json (verified during
        // fixture generation).
        for (text, id) in [
            ("<|system|>", 154826u32),
            ("<|user|>", 154827),
            ("<|assistant|>", 154828),
            ("<|observation|>", 154829),
            ("[gMASK]", 154822),
            ("<sop>", 154824),
            ("<think>", 154841),
            ("</think>", 154842),
            ("<tool_call>", 154843),
            ("<tool_response>", 154845),
            ("<arg_key>", 154847),
            ("<arg_value>", 154849),
        ] {
            let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
            assert_eq!(ids, vec![id], "special token {text}");
        }
    }

    #[test]
    fn plain_english_count_is_stable() {
        let ids = encode_no_special("hello world").unwrap();
        assert_eq!(ids.len(), 2);
    }
}
