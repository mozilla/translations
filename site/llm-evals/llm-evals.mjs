/**
 * @import { Evaluation, ScoreType, Analysis, ScoreNumbers, Summary } from "./llm-evals"
 */

import {
  getElement,
  exposeAsGlobal,
  create,
  changeLocation,
} from "../utils.mjs";
import { terminology } from "./terminology.mjs";

const elements = {
  loading: getElement("loading"),
  error: getElement("error"),
  tbody: getElement("tbody"),
  table: getElement("table"),
  baseUrl: /** @type {HTMLInputElement} */ (getElement("baseUrl")),
  form: getElement("form"),
};

/**
 * @param {string} message
 */
function showError(message) {
  elements.error.style.display = "block";
  elements.error.innerText = message;
  elements.loading.style.display = "none";
}

/**
 * @returns {null | string}
 */
function getBaseUrl() {
  const urlParams = new URLSearchParams(window.location.search);
  const baseUrl = urlParams.get("baseUrl");
  if (!baseUrl) {
    return null;
  }
  // Validate that it's a URL.
  try {
    new URL(baseUrl);
  } catch {
    return null;
  }
  if (!baseUrl.endsWith("/") && !baseUrl.endsWith("%2F")) {
    return baseUrl + "/";
  }
  return baseUrl;
}

document.addEventListener("DOMContentLoaded", async () => {
  const baseUrl = getBaseUrl();
  if (baseUrl) {
    initializePage(baseUrl);
    elements.baseUrl.value = baseUrl;
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const newBaseUrl = elements.baseUrl.value;
    try {
      // Validate the URL.
      new URL(newBaseUrl);
    } catch {
      showError("The location was not a valid URL.");
      return;
    }
    const urlParams = new URLSearchParams();
    urlParams.set("baseUrl", newBaseUrl);
    changeLocation(urlParams);
  });
});

/**
 * @param {string} baseUrl
 */
async function initializePage(baseUrl) {
  elements.loading.style.display = "block";

  /** @type {Evaluation[]} */
  let evals;
  try {
    evals = await getEvals(baseUrl);
  } catch (error) {
    showError("Failed to get the evals. See the console for more information.");
    console.error(error);
    return;
  }
  /** @type {Summary} */
  let summary;
  try {
    summary = await getSummary(baseUrl);
  } catch (error) {
    showError(
      "Failed to get the summary. See the console for more information."
    );
    console.error(error);
    return;
  }

  elements.loading.style.display = "none";
  elements.table.style.display = "table";

  exposeAsGlobal("evals", evals);
  const analysis = analyzeEvals(evals);
  exposeAsGlobal("analysis", analysis);

  renderAnalysis(evals, analysis, summary);
}

/**
 * @param {string} baseUrl
 * @returns {Promise<Evaluation[]>}
 */
async function getEvals(baseUrl) {
  const response = await fetch(baseUrl + "scores.json");
  return response.json();
}

/**
 * @param {string} baseUrl
 * @returns {Promise<Summary>}
 */
async function getSummary(baseUrl) {
  const response = await fetch(baseUrl + "summary.json");
  return response.json();
}

/**
 * Compute mean, median, and histogram for evaluation scores.
 *
 * @param {Evaluation[]} evals
 * @returns {Record<ScoreType, Analysis>}
 */
function analyzeEvals(evals) {
  /** @type {Record<ScoreType, ScoreNumbers[]>} */
  const scoresByType = {
    adequacy: [],
    fluency: [],
    terminology: [],
    hallucination: [],
    punctuation: [],
  };

  // Group the scores by type.
  for (const evaluation of evals) {
    if (!evaluation.scores) {
      continue;
    }
    for (const key of /** @type {(ScoreType)[]} */ (
      Object.keys(evaluation.scores)
    )) {
      const [score] = evaluation.scores[key];
      scoresByType[key].push(score);
    }
  }

  /**
   * @type {Record<ScoreType, Analysis>}
   */
  return {
    adequacy: summarize(scoresByType.adequacy),
    fluency: summarize(scoresByType.fluency),
    terminology: summarize(scoresByType.terminology),
    hallucination: summarize(scoresByType.hallucination),
    punctuation: summarize(scoresByType.punctuation),
  };
}

/**
 * Summarize a numeric array.
 *
 * @param {ScoreNumbers[]} values
 * @returns {Analysis}
 */
function summarize(values) {
  /** @type {Record<ScoreNumbers, number>} */
  const histogram = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 };

  for (const v of values) {
    histogram[v]++;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;

  return { mean, median, histogram };
}

/**
 * @param {Evaluation[]} evals
 * @param {Record<ScoreType, Analysis>} analysis
 * @param {Summary} summary
 */
function renderAnalysis(evals, analysis, summary) {
  for (const [scoreTypeStr, data] of Object.entries(analysis)) {
    const scoreType = /** @type {ScoreType} */ (scoreTypeStr);
    const { description, scales } = terminology[scoreType];
    const examplesRow = create.tr({
      className: "examplesRow",
    });
    create.tr({
      parent: elements.tbody,
      className: "criteria",
      children: [
        create.td({
          children: [
            //
            create.h3({ children: scoreType }),
            create.p({
              className: "criteraDescription",
              children: description,
            }),
          ],
        }),
        create.td({
          children: `${data.mean.toFixed(2)} – ${summary[scoreType]}`,
        }),
        create.td({
          children: [
            //
            data.median,
            " – ",
            scales[data.median],
          ],
        }),
        create.td({
          children: createHistogram(
            evals,
            scoreType,
            data.histogram,
            scales,
            examplesRow
          ),
        }),
      ],
    });

    elements.tbody.appendChild(examplesRow);
  }
}

/**
 * @param {Evaluation[]} evals
 * @param {ScoreType} scoreType
 * @param {Record<number, number>} histogram
 * @param {Record<ScoreNumbers, string>} scales - The description of the scales.
 * @param {HTMLTableRowElement} examplesRow
 */

function createHistogram(evals, scoreType, histogram, scales, examplesRow) {
  const labels = [1, 2, 3, 4, 5];
  const values = labels.map((k) => histogram[k]);
  const total = values.reduce((sum, v) => sum + v, 0);

  const clearRow = () => {
    while (examplesRow.firstChild) {
      examplesRow.firstChild.remove();
    }
  };

  return create.div({
    className: "histogram",
    children: labels.map((label, index) => {
      const count = values[index];
      const freq = total > 0 ? count / total : 0;
      const score = /** @type {ScoreNumbers} */ (index + 1);
      const scoreDocumentation = scales[score];

      return create.button({
        className: "histogramBucket",
        onClick() {
          clearRow();

          // Programatically determine the colspan of the previous row.
          const previousTR = /** @type {HTMLElement} */ (
            examplesRow.previousElementSibling
          );
          const colspan = [...previousTR.querySelectorAll("td")].length;

          create.td({
            parent: examplesRow,
            attrs: { colspan },
            children: [
              create.div({
                className: "examplesHeader",
                children: create.button({
                  attrs: { type: "button" },
                  children: "Close",
                  onClick() {
                    clearRow();
                  },
                }),
              }),
              create.div({
                className: "examples",
                children: evals
                  .filter(({ scores }) => scores[scoreType][0] === label)
                  .map(({ translation, scores }) => {
                    const { src, trg, ref } = translation;
                    const [_value, description] = scores[scoreType];
                    return create.div({
                      className: "example",
                      children: [
                        create.div({
                          className: "exampleDescription",
                          children: [
                            create.span({ children: "llm: " }),
                            description,
                          ],
                        }),
                        create.div({
                          className: "exampleSentences",
                          children: [
                            create.div({
                              children: [
                                create.div({
                                  className: "exampleSrc",
                                  children: [
                                    create.span({ children: "src: " }),
                                    src,
                                  ],
                                }),
                              ],
                            }),
                            create.div({
                              children: [
                                create.div({
                                  className: "exampleTrg",
                                  children: [
                                    create.span({ children: "trg: " }),
                                    trg,
                                  ],
                                }),
                                create.div({
                                  className: "exampleRef",
                                  children: [
                                    create.span({ children: "ref: " }),
                                    ref,
                                  ],
                                }),
                              ],
                            }),
                          ],
                        }),
                      ],
                    });
                  }),
              }),
            ],
          });

          examplesRow.style.display = "table-row";
        },
        title: `${score} – ${scoreDocumentation}`,
        children: [
          create.div({
            className: "histogramCount",
            children: count,
          }),
          create.div({
            className: "histogramBar",
            style: { height: `${freq * 100}px` },
          }),
          create.div({
            className: "histogramBucketNumber",
            children: `${label}`,
          }),
        ],
      });
    }),
  });
}
