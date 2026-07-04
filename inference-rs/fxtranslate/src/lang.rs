//! Static language-tag → display-name mapping.
//!
//! Names are taken from Google Cloud Translation's supported-languages list
//! (<https://docs.cloud.google.com/translate/docs/languages>), keyed by the tags
//! Firefox Translations actually ships (all 54 as of writing). Two tags differ
//! from Google's own codes and are mapped by language: `zh-Hans`/`zh-Hant` →
//! Google's Chinese (Simplified)/(Traditional) (`zh-CN`/`zh-TW`). `nn` (Norwegian
//! Nynorsk) is not on Google's page, so it falls back to the code (see
//! [`display_name`]). Deliberately a small static table, not a display-names
//! library.

/// `(tag, display name)`, sorted by tag. Every entry here is present in Google's
/// list except the two Chinese tags, which are mapped by language identity.
const NAMES: &[(&str, &str)] = &[
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("bs", "Bosnian"),
    ("ca", "Catalan"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("de", "German"),
    ("el", "Greek"),
    ("en", "English"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("gu", "Gujarati"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("id", "Indonesian"),
    ("is", "Icelandic"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kn", "Kannada"),
    ("ko", "Korean"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("ml", "Malayalam"),
    ("ms", "Malay"),
    ("nb", "Norwegian Bokmål"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sq", "Albanian"),
    ("sr", "Serbian"),
    ("sv", "Swedish"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("vi", "Vietnamese"),
    ("zh-Hans", "Chinese (Simplified)"),
    ("zh-Hant", "Chinese (Traditional)"),
];

/// The display name for a language tag, or the tag itself if unknown (e.g. `nn`).
pub fn display_name(tag: &str) -> &str {
    NAMES
        .iter()
        .find(|(t, _)| *t == tag)
        .map(|(_, name)| *name)
        .unwrap_or(tag)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_tags_map_to_names() {
        assert_eq!(display_name("es"), "Spanish");
        assert_eq!(display_name("en"), "English");
        assert_eq!(display_name("zh-Hans"), "Chinese (Simplified)");
        assert_eq!(display_name("zh-Hant"), "Chinese (Traditional)");
        assert_eq!(display_name("fa"), "Persian");
    }

    #[test]
    fn unknown_tag_falls_back_to_code() {
        // `nn` (Norwegian Nynorsk) is not on Google's page.
        assert_eq!(display_name("nn"), "nn");
        assert_eq!(display_name("xx"), "xx");
    }

    #[test]
    fn table_is_sorted_and_unique() {
        for w in NAMES.windows(2) {
            assert!(
                w[0].0 < w[1].0,
                "NAMES must be sorted/unique: {} !< {}",
                w[0].0,
                w[1].0
            );
        }
    }
}
