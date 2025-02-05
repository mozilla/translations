// Enables setting runtime args via the query string (as long as this is run in the main browser thread and not in a worker)
let stdinInput = false;
if (
  typeof window !== "undefined" &&
  window.location &&
  window.location.search
) {
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.get("stdinInput")) {
    stdinInput = urlParams.get("stdinInput");
    console.log("Using stdinInput from URL");
  }
  if (urlParams.get("arguments")) {
    Module["arguments"] = urlParams.get("arguments").split(' ')
    // Module["arguments"] = urlParams.get("arguments").split('%20');
    console.log("Using arguments from URL");
  }
}
console.log('stdinInput', stdinInput);
console.log('Module["arguments"]', Module["arguments"]);
Module["noInitialRun"] = true;
Module["onRuntimeInitialized"] = _ => {
  try {
    console.log("Calling main in a try-catch block to be able to get readable exception messages");
    callMain(Module["arguments"])
  } catch (exception) {
    console.error("WASM exception thrown", Module.getExceptionMessage(exception))
  }
};
var initStdInOutErr = function() {
  var i = 0;
  function stdin() {
    if (stdinInput === false) {
      console.log("STDIN: No stdin input specified");
      return null;
    }
    var input = stdinInput + "\n";
    if (i < input.length) {
      var code = input.charCodeAt(i);
      ++i;
      console.log("STDIN: Feeding character code to stdin: ", code);
      return code;
    } else {
      console.log("STDIN: Done feeding input via stdin: ", input);
      return null;
    }
  }

  var stdoutBuffer = "";
  function stdout(code) {
    if (code === "\n".charCodeAt(0) && stdoutBuffer !== "") {
      console.log("STDOUT: ", stdoutBuffer);
      stdoutBuffer = "";
    } else {
      stdoutBuffer += String.fromCharCode(code);
    }
  }

  var stderrBuffer = "";
  function stderr(code) {
    if (code === "\n".charCodeAt(0) && stderrBuffer !== "") {
      console.log("STDERR: ", stderrBuffer);
      stderrBuffer = "";
    } else {
      stderrBuffer += String.fromCharCode(code);
    }
  }

  FS.init(stdin, stdout, stderr);
}
Module["preRun"].push(initStdInOutErr);
