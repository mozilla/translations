PONTOON_LANGUAGES = { "aa", "aat", "ab", "abb", "abq", "ace", "ach", "ady", "af", "ajg", "ak",
    "aln", "am", "an", "ann", "anp", "ar", "arn", "as", "ast", "ay", "az", "azb", "azz",
    "ba", "bag", "bal", "ban", "bas", "bax", "bba", "bbj", "bbl", "bce", "bci", "be", "beb",
    "bew", "bfd", "bft", "bg", "bgp", "bkh", "bkm", "bm", "bn", "bnm", "bnn", "bo", "bqi",
    "br", "brh", "bri", "brx", "bs", "bsh", "bsk", "bsy", "btv", "bum", "bxk", "bxr", "byv",
    "ca", "cak", "cdo", "ceb", "cgg", "cjk", "ckb", "cnh", "co", "cpx", "cpy", "crh", "cs",
    "csb", "cut", "cux", "cv", "cy", "da", "dag", "dar", "dav", "de", "din", "dmk", "dml",
    "dru", "dsb", "dua", "dv", "dyu", "ebr", "ee", "eko", "el", "en", "eo", "es", "esu",
    "et", "eto", "eu", "ewo", "fa", "fan", "ff", "fi", "fmp", "fo", "fr", "frp", "fub",
    "fue", "fuf", "fur", "fy", "ga", "gaa", "gd", "gej", "ggg", "gid", "gig", "giz", "gjk",
    "gju", "gl", "gn", "gom", "gor", "gos", "gsw", "gu", "guc", "gv", "gwc", "gwt", "gya",
    "ha", "hac", "haz", "hch", "he", "hem", "hi", "hil", "hno", "hr", "hrx", "hsb", "ht",
    "hu", "hus", "hux", "hy", "hye", "hyw", "ia", "iba", "ibb", "id", "ie", "ig", "ilo",
    "ipk", "is", "it", "ixl", "izh", "ja", "jam", "jbo", "jgo", "jiv", "jqr", "jv", "ka",
    "kaa", "kab", "kam", "kbd", "kcn", "kdh", "khw", "ki", "kk", "kln", "kls", "km", "kmr",
    "kn", "knn", "ko", "kok", "koo", "kpv", "krc", "ks", "ksf", "kvx", "kw", "kxp", "ky",
    "kzi", "lb", "led", "leu", "lg", "lij", "lke", "lld", "ln", "lo", "lrk", "lrl", "lss",
    "lt", "ltg", "lth", "lua", "luo", "lus", "lv", "lzz", "mai", "mau", "mbf", "mbo", "mcf",
    "mcn", "mcx", "mdd", "mdf", "meh", "mel", "mfe", "mg", "mgg", "mhk", "mhr", "mix", "mk",
    "mki", "ml", "mmc", "mn", "mni", "mos", "mqh", "mr", "mrh", "mrj", "ms", "mse", "msi",
    "mt", "mua", "mug", "mve", "mvy", "mxu", "my", "myv", "nan", "nb", "ncx", "nd", "ne",
    "new", "nhe", "nhi", "nia", "nl", "nla", "nlv", "nmg", "nmz", "nn", "nnh", "nqo", "nr",
    "nso", "nv", "ny", "nyn", "nyu", "oc", "odk", "om", "or", "oru", "os", "pa", "pai",
    "pap", "pcd", "pcm", "pez", "phl", "phr", "pl", "plk", "pne", "ppl", "prq", "ps", "pt",
    "pua", "pwn", "quc", "qug", "qup", "qur", "qus", "qux", "quy", "qva", "qvi", "qvj", "qvl",
    "qwa", "qws", "qxa", "qxp", "qxq", "qxt", "qxu", "qxw", "rif", "rm", "rn", "ro", "rof",
    "ru", "ruc", "rup", "rw", "rwm", "sah", "sat", "sbn", "sc", "scl", "scn", "sco", "sd",
    "sdh", "sdo", "seh", "sei", "ses", "shi", "shn", "si", "sk", "skr", "sl", "sn", "snk",
    "snv", "so", "son", "sq", "sr", "ss", "ssi", "st", "su", "sv", "sva", "sw", "syr", "szl",
    "szy", "ta", "tar", "tay", "te", "teg", "tg", "th", "ti", "tig", "tk", "tl", "tli", "tn",
    "tob", "tok", "top", "tr", "trs", "trv", "trw", "ts", "tsz", "tt", "ttj", "tui", "tvu",
    "tw", "ty", "tyv", "tzm", "uby", "udl", "udm", "ug", "uk", "ukv", "ur", "ush", "uz", "var",
    "ve", "vec", "vi", "vmw", "vot", "wbl", "wep", "wes", "wo", "xcl", "xdq", "xh", "xhe",
    "xka", "xkl", "xmf", "xsm", "yaq", "yav", "ydg", "yi", "yo", "yua", "yue", "zam", "zgh",
    "zh", "zoc", "zu", "zza"
}  # fmt: skip


def pontoon_handle_bcp(lang):
    if lang == "sv":
        return "sv-SE"
    if lang == "gu":
        return "gu-IN"
    if lang == "pa":
        return "pa-IN"
    if lang == "nn":
        return "nn-NO"
    if lang == "nb":
        return "nb-NO"
    if lang == "no":
        return "nb-NO"
    if lang == "ne":
        return "ne-NP"
    if lang == "hi":
        return "hi-IN"
    if lang == "hy":
        return "hy-AM"
    if lang == "ga":
        return "ga-IE"
    if lang == "bn":
        return "bn-IN"
    if lang == "zh":
        return "zh-CN"
    return lang
