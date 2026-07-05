//! Binary entry point. The CLI logic — parsing, dispatch, `list`/`translate` —
//! lives in the library so it's testable without a network or engine; `main`
//! only supplies the real ones (plus the terminal).

use std::io::{BufReader, IsTerminal};
use std::process::ExitCode;

use fxtranslate_cli::cli::{self, Deps, Io};
use fxtranslate_cli::fetch::NetworkFetch;
use fxtranslate_cli::translate::EngineTranslator;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let stdin_is_tty = std::io::stdin().is_terminal();
    let stdout_is_tty = std::io::stdout().is_terminal();
    // Download progress goes to stderr, so gate it on stderr being a terminal.
    let stderr_is_tty = std::io::stderr().is_terminal();

    let fetch = NetworkFetch::new();
    let translator = EngineTranslator::new(&fetch, stderr_is_tty);
    let deps = Deps {
        fetch: &fetch,
        translator: &translator,
    };

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
