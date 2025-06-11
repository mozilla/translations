// @ts-check
import { changeLocation, exposeAsGlobal, getElement } from "../utils.mjs";

/**
 * @import { ModelRecord, EvalResults } from "../@types/models"
 */

main().catch((error) => {
  console.error(error);
  getElement("error").style.display = "block";
});

const aLessThanB = "a".localeCompare("b");
const aGreaterThanB = aLessThanB * -1;
const aEqualToB = 0;

/**
 * @param {string} url
 * @returns {Promise<any>}
 */
async function fetchJSON(url) {
  const response = await fetch(url);

  if (!response.ok) {
    console.error(response);
    throw new Error("Response failed.");
  }

  return await response.json();
}

async function main() {
  getElement("counts").style.display = "table";

  const remoteSettingsPreviewCheckbox = /** @type {HTMLInputElement} */ (
    getElement("remoteSettingsPreview")
  );
  const urlParams = new URLSearchParams(window.location.search);
  const isPreview = urlParams.get("preview");
  remoteSettingsPreviewCheckbox.checked = isPreview === "true";
  remoteSettingsPreviewCheckbox.addEventListener("change", () => {
    const urlParams = new URLSearchParams(window.location.search);
    if (remoteSettingsPreviewCheckbox.checked) {
      urlParams.set("preview", "true");
    } else {
      urlParams.delete("preview");
    }
    changeLocation(urlParams);
  });

  const bucket = isPreview ? "main-preview" : "main";

  /** @type {{ data: ModelRecord[] }} */
  const records = await fetchJSON(
    `https://firefox.settings.services.mozilla.com/v1/buckets/${bucket}/collections/translations-models/records`
  );
  exposeAsGlobal("records", records.data);

  const attachmentsByKey = getAttachmentsByKey(records.data);
  countModels(records.data);

  /** @type {EvalResults} */
  const cometResults = await fetchJSON(
    "https://raw.githubusercontent.com/mozilla/firefox-translations-models/main/evaluation/comet-results.json"
  );

  logCometResults(cometResults);

  /**
   * @typedef {Object} ModelEntry
   * @property {string} lang
   * @property {string} version
   * @property {string} display
   * @property {ModelRecord[]} fromEn
   * @property {ModelRecord[]} toEn
   */

  /** @type {Map<string, ModelEntry>} */
  const modelsMap = new Map();
  const models = records.data.filter((record) => record.fileType === "model");
  exposeAsGlobal("models", models);

  const dn = new Intl.DisplayNames("en", {
    type: "language",
    fallback: "code",
    languageDisplay: "standard",
  });

  for (const model of models) {
    /** @type {ModelEntry | undefined} */
    let entry;
    if (model.fromLang === "en") {
      entry = modelsMap.get(model.toLang + " " + model.version);
      if (!entry) {
        entry = {
          lang: model.toLang,
          version: model.version,
          display: dn.of(model.toLang) ?? model.toLang,
          toEn: [],
          fromEn: [],
        };
      }
      entry.fromEn.push(model);
    } else {
      entry = modelsMap.get(model.fromLang + " " + model.version);
      if (!entry) {
        entry = {
          lang: model.fromLang,
          version: model.version,
          display: dn.of(model.fromLang) ?? model.fromLang,
          toEn: [],
          fromEn: [],
        };
      }
      entry.toEn.push(model);
    }
    modelsMap.set(entry.lang + " " + model.version, entry);
  }

  const tbody = getElement("tbody");

  const modelEntries = [...modelsMap.values()].sort((a, b) =>
    `${a.lang}`.localeCompare(b.lang)
  );
  modelEntries.sort((a, b) => a.display.localeCompare(b.display));

  for (const entry of modelEntries) {
    entry.fromEn.sort((a, b) => -versionCompare(a.version, b.version));
    entry.toEn.sort((a, b) => -versionCompare(a.version, b.version));
  }

  for (const { lang, toEn, fromEn } of modelEntries) {
    const tr = document.createElement("tr");
    /**
     * @param {string} [text]
     */
    const td = (text = "") => {
      const el = document.createElement("td");
      el.innerText = text;
      tr.appendChild(el);
      return el;
    };
    td(dn.of(lang));

    addToRow(
      td,
      `${lang}-en`,
      records.data,
      cometResults,
      attachmentsByKey,
      toEn[0]
    );
    addToRow(
      td,
      `en-${lang}`,
      records.data,
      cometResults,
      attachmentsByKey,
      fromEn[0]
    );
    tbody.append(tr);
  }
  getElement("loading").style.display = "none";
  getElement("table").style.display = "table";
}

/**
 * @param {(text?: string) => HTMLTableCellElement} td
 * @param {string} pair
 * @param {ModelRecord[]} records
 * @param {EvalResults} cometResults
 * @param {Map<string, Array<[string, string]>>} attachmentsByKey
 * @param {ModelRecord} [model]
 */
function addToRow(td, pair, records, cometResults, attachmentsByKey, model) {
  const modelNameTD = td();
  if (model) {
    // Add the attachments.
    const attachments = attachmentsByKey.get(getAttachmentKey(model));
    if (attachments) {
      const div = document.createElement("div");
      div.className = "attachments";
      for (const [name, url] of attachments) {
        const a = document.createElement("a");
        a.innerText = name;
        a.href = url;
        div.appendChild(a);
      }
      const button = document.createElement("button");
      button.innerText = pair;

      // Hide when clicking outside of the button and popup.
      document.body.addEventListener("click", (event) => {
        const target = /** @type {Node | null} */ (event.target);
        if (target && !div.contains(target) && target !== button) {
          div.style.display = "none";
        }
      });

      button.addEventListener("click", () => {
        if (div.style.display === "block") {
          div.style.display = "none";
        } else {
          div.style.display = "block";
        }
      });

      modelNameTD.appendChild(button);
      modelNameTD.appendChild(div);
    } else {
      modelNameTD.innerText = pair;
    }
  }

  td(model?.version);
  td(getModelSize(records, model));

  const releaseEl = td(getReleaseChannel(model));
  releaseEl.title = model?.filter_expression ?? "";

  const googleComet = cometResults[pair]?.["flores-test"]?.["google"];
  const bergamotComet = cometResults[pair]?.["flores-test"]?.["bergamot"];
  const googleCometAvg = getAverageScore(pair, cometResults, "google");
  const bergamotCometAvg = getAverageScore(pair, cometResults, "bergamot");

  const hasEvals = bergamotComet && googleComet;

  const bergamotCometDisplay = (100 * bergamotComet).toFixed(2);
  const percentage = 100 * (1 - googleComet / bergamotComet);
  const sign = percentage >= 0 ? "+" : "";
  let scoreDisplay = "";
  if (hasEvals) {
    const percentDisplay = `${sign}${percentage.toFixed(2)}%`.padStart(
      7,
      "\u00A0"
    );
    scoreDisplay = `${bergamotCometDisplay} ${percentDisplay}`;
  }
  const avgPercentage = 100 * (1 - googleCometAvg / bergamotCometAvg);
  const avgSign = avgPercentage >= 0 ? "+" : "";
  const avgPercentageDisplay = hasEvals
    ? `${avgSign}${avgPercentage.toFixed(2)}%`
    : "";

  const el = td(scoreDisplay);
  if (hasEvals) {
    let shippable = "Shippable";
    // el.style.color = "#fff";
    // el.style.background = "#2ebffc";
    if (percentage < -5) {
      // Does not meet release criteria.
      el.style.background = "#ffa537";
      // el.style.color = "#000";
      shippable = "Not shippable";
    }

    el.title =
      `${shippable} - COMET ${(100 * bergamotComet).toFixed(2)} ` +
      `vs Google Comet ${(100 * googleComet).toFixed(2)} ` +
      `(${scoreDisplay})` +
      "\n\n" +
      `avg COMET ${(100 * bergamotCometAvg).toFixed(2)} ` +
      `vs Google avg Comet ${(100 * googleCometAvg).toFixed(2)} ` +
      `(${avgPercentageDisplay})`;
  }
}

/**
 * @param {string} pair
 * @param {EvalResults} cometResults
 * @param {string} translator
 */
function getAverageScore(pair, cometResults, translator) {
  let count = 0;
  let total = 0;
  const datasets = cometResults[pair];
  if (!datasets) {
    return 0;
  }
  for (const obj of Object.values(datasets)) {
    const score = obj[translator];
    if (score) {
      count++;
      total += score;
    }
  }
  if (count === 0) {
    return 0;
  }
  return total / count;
}

/**
 * @param {ModelRecord[]} records
 * @param {ModelRecord} [model]
 */
function getModelSize(records, model) {
  if (!model) {
    return "";
  }

  let size = 0;
  for (const record of records) {
    if (
      record.fromLang === model.fromLang &&
      record.toLang === model.toLang &&
      record.version === model.version &&
      record.filter_expression === model.filter_expression
    ) {
      size += Number(record.attachment.size);
    }
  }

  return (size / 1000 / 1000).toFixed(1) + " MB";
}

/**
 * Compare two versions quickly.
 * @param {string} a
 * @param {string} b
 * @return {number}
 */
export default function versionCompare(a, b) {
  /** @type {any[]} */
  const aParts = a.split(".");
  /** @type {any[]} */
  const bParts = b.split(".");
  while (aParts.length < 3) {
    aParts.unshift("0");
  }
  while (bParts.length < 3) {
    bParts.unshift("0");
  }

  const [, aEnd, aBeta] = aParts[2].match(/(\d+)([a-z]\d?)?/) ?? [
    undefined,
    "0",
    "",
  ];
  const [, bEnd, bBeta] = bParts[2].match(/(\d+)([a-z]\d?)?/) ?? [
    undefined,
    "0",
    "",
  ];
  aParts.pop();
  bParts.pop();
  aParts.push(aEnd);
  bParts.push(bEnd);

  aParts[0] = Number(aParts[0]);
  aParts[1] = Number(aParts[1]);
  aParts[2] = Number(aParts[2]);

  bParts[0] = Number(bParts[0]);
  bParts[1] = Number(bParts[1]);
  bParts[2] = Number(bParts[2]);

  for (const part of aParts) {
    if (isNaN(part)) {
      console.error(aParts);
      throw new Error(a + " had an NaN.");
    }
  }
  for (const part of bParts) {
    if (isNaN(part)) {
      console.error(bParts);
      throw new Error(a + " had an NaN.");
    }
  }

  for (let i = 0; i < 3; i++) {
    const aPart = aParts[i];
    const bPart = bParts[i];
    if (aPart > bPart) return aGreaterThanB;
    if (aPart < bPart) return aLessThanB;
  }
  if (!aBeta && !bBeta) return aEqualToB;
  if (!aBeta) return aGreaterThanB;
  if (!bBeta) return aLessThanB;

  return aBeta.localeCompare(bBeta);
}

/**
 * @param {ModelRecord} [model]
 * @returns {string}
 */
function getReleaseChannel(model) {
  if (!model) {
    return "";
  }
  let filterExpression = model.filter_expression ?? "";
  filterExpression = filterExpression.replace(
    "env.channel == 'default'",
    "Local Build"
  );
  filterExpression = filterExpression.replace(
    "env.channel == 'nightly'",
    "Nightly"
  );
  filterExpression = filterExpression.replace("env.channel == 'beta'", "Beta");
  filterExpression = filterExpression.replace(
    "env.channel == 'release'",
    "Release"
  );
  filterExpression = filterExpression.replace(
    "env.channel == 'aurora'",
    "Aurora"
  );
  filterExpression = filterExpression.replace("||", "or");
  filterExpression = filterExpression.replace("&&", "and");
  if (!filterExpression) {
    return "Released";
  }
  if (model.version?.endsWith("a1")) {
    filterExpression = "Local Build or Nightly";
  }

  if (filterExpression === "Local Build or Nightly") {
    // Simplify this to just nightly.
    return "Nightly";
  }
  return "Custom";
}

/**
 * @param {string} a
 * @param {string} b
 * @param {number} direction
 */
function assertComparison(a, b, direction) {
  if (versionCompare(a, b) !== direction) {
    throw new Error(`Expected ${a} ${b} to compare to ${direction}`);
  }
}

assertComparison("1.0a", "1.0", aLessThanB);
assertComparison("1.0a1", "1.0", aLessThanB);
assertComparison("1.0a", "1.0a", aEqualToB);
assertComparison("0.1.0a", "1.0a", aEqualToB);
assertComparison("1.0", "1.0a", aGreaterThanB);
assertComparison("1.0", "1.0a1", aGreaterThanB);
assertComparison("1.0", "2.0", aLessThanB);
assertComparison("1.0", "1.1", aLessThanB);
assertComparison("1.0a", "1.1", aLessThanB);

/**
 * @param {EvalResults} cometResults
 */
function logCometResults(cometResults) {
  /** @type {Array<unknown[]>} */
  const xx_en = [];
  const en_xx = [];

  for (const [langPair, evaluation] of Object.entries(cometResults)) {
    const flores = evaluation["flores-dev"];
    const [fromLang, toLang] = langPair.split("-");
    const row = [
      langPair,
      fromLang,
      toLang,
      flores.google || "",
      flores.bergamot || "",
    ];
    if (fromLang === "en") {
      en_xx.push(row);
    } else {
      xx_en.push(row);
    }
  }

  /**
   * @param {any} a
   * @param {any} b
   */
  function sortRow(a, b) {
    return (a[1] + "-" + a[2]).localeCompare(b[1] + "-" + b[2]);
  }
  xx_en.sort(sortRow);
  en_xx.sort(sortRow);

  const rows = [
    ["Lang Pair", "From", "To", "Google", "Bergamot"],
    ...en_xx,
    ...xx_en,
  ];

  let tsv = "";
  for (const row of rows) {
    tsv += row.join("\t") + "\n";
  }

  console.log(tsv);
}

/**
 * @param {ModelRecord} record
 */
function getAttachmentKey(record) {
  const { fromLang, toLang, version } = record;
  return `${fromLang}-${toLang} ${version}`;
}

/**
 * @param {ModelRecord[]} records
 */
function getAttachmentsByKey(records) {
  /** @type {Map<string, Array<[string, string]>>} */
  const attachmentsByKey = new Map();
  for (const record of records) {
    const key = getAttachmentKey(record);
    let attachments = attachmentsByKey.get(key);
    if (!attachments) {
      attachments = [];
      attachmentsByKey.set(key, attachments);
    }
    attachments.push([
      record.name,
      `https://firefox-settings-attachments.cdn.mozilla.net/${record.attachment.location}`,
    ]);
  }
  return attachmentsByKey;
}

/**
 * @param {ModelRecord[]} records
 */
function countModels(records) {
  const fromProd = new Set();
  const fromNightly = new Set();
  const toProd = new Set();
  const toNightly = new Set();

  for (const record of records) {
    const isRelease =
      !record.filter_expression ||
      record.filter_expression.includes("env.channel == 'release'");
    if (record.fromLang == "en") {
      if (isRelease) {
        toProd.add(record.toLang);
      } else {
        toNightly.add(record.toLang);
      }
    } else {
      if (isRelease) {
        fromProd.add(record.fromLang);
      } else {
        fromNightly.add(record.fromLang);
      }
    }
  }

  const toNightlyOnly = toNightly.difference(toProd);
  const fromNightlyOnly = fromNightly.difference(fromProd);

  getElement("fromProd").innerText = String(fromProd.size);
  getElement("toProd").innerText = String(toProd.size);
  getElement("fromNightly").innerText = String(toNightlyOnly.size);
  getElement("toNightly").innerText = String(fromNightlyOnly.size);
}
