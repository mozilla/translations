//! Binary entry point. The CLI logic — parsing, dispatch, `list`/`translate` —
//! lives in the library so it's testable without a network or engine; `main`
//! only supplies the real ones (plus the terminal).

use std::io::{BufReader, IsTerminal};
use std::process::ExitCode;

use fxtranslate::cli::{self, Deps, Io};
use fxtranslate::fetch::NetworkFetch;
use fxtranslate::translate::EngineTranslator;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let fetch = NetworkFetch;
    let translator = EngineTranslator::new(&fetch);
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };

    let stdin_is_tty = std::io::stdin().is_terminal();
    let stdout_is_tty = std::io::stdout().is_terminal();
    let no_color = std::env::var_os("NO_COLOR").is_some();

    let mut stdin = BufReader::new(std::io::stdin());
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    let mut io = Io {
        stdin: &mut stdin,
        stdout: &mut stdout,
        stderr: &mut stderr,
        stdin_is_tty,
        stdout_is_tty,
        no_color,
    };

    cli::run(&args, &deps, &mut io)
}
