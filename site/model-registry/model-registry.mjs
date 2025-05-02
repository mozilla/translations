// @ts-check
import {
  create,
  exposeAsGlobal,
  formatBytes,
  getElement,
  isNever,
  jsonToYAML,
  parseSearchQuery,
} from "../utils.mjs";

/**
 * @import {
 *   Corpus,
 *   TrainingRun,
 *   ModelRun,
 *   ModelReference,
 *   ModelName
 * } from '../@types/training-run.d.ts'
 */

const BUCKET_NAME = "moz-fx-translations-data--303e-prod-translations-data";
const STORAGE_URL = `https://storage.googleapis.com/${BUCKET_NAME}`;

/**
 * The elements for the page get selected here in a type-friendly manner. If the elements
 * aren't found, then there is a runtime error.
 */
const elements = {
  table: getElement("table", HTMLTableElement),
  tbody: getElement("table-body"),
  thead: getElement("table-thead"),
  tableContainer: getElement("table-container", HTMLDivElement),
  loading: getElement("loading"),
  error: getElement("error"),
  searchFilter: getElement("search-filter", HTMLInputElement),
  overlay: getElement("overlay"),
  overlayCloseButton: getElement("overlay-close-button"),
  overlayContent: getElement("overlay-content"),
  scrollContainer: getElement("scroll-container"),
  scores: getElement("scores"),
};

/**
 * The URL-serialized state as inferred by:
 * @see {URLStateManager.prototype.getInitialState}
 *
 * @typedef {ReturnType<typeof URLStateManager.prototype.getInitialState>} State
 */

/**
 * This is a helper clss that manages the state of the view that is URL serialized
 * and pushed onto the history.
 */
class URLStateManager {
  /**
   * @type {State}
   */
  state = this.getInitialState();

  constructor() {
    addEventListener("popstate", (event) => {
      this.state = event.state.state;
      this.updateUI();
    });
  }

  /**
   * Initializes the current {@link State} from the URLParams.
   */
  getInitialState() {
    const urlParams = new URLSearchParams(window.location.search);

    /** @type {ModelReference | null} */
    let modelReference = null;
    {
      const name = urlParams.get("modelName");
      const langpair = urlParams.get("modelLangpair");
      const modelName = toModelName(urlParams.get("modelModelName"));

      if (name && langpair && modelName) {
        modelReference = {
          name,
          langpair,
          modelName,
        };
      }
    }

    // The types for the State are inferred from this return:
    return {
      searchString: urlParams.get("searchString") ?? "",
      showModels: urlParams.get("showModels") == "true" ? true : false,
      showCorpora: urlParams.get("showCorpora") == "true" ? true : false,
      score: urlParams.get("score") || "vs-google",
      modelReference,
    };
  }

  /**
   * Converts the {@link State} URLSearchParams.
   * @returns {URLSearchParams}
   */
  stateToURLSearchParams() {
    const urlParams = new URLSearchParams();
    urlParams.set("searchString", this.state.searchString);
    if (this.state.showModels) {
      urlParams.set("showModels", "true");
    }
    if (this.state.showCorpora) {
      urlParams.set("showCorpora", "true");
    }
    if (this.state.modelReference) {
      urlParams.set("modelName", this.state.modelReference.name);
      urlParams.set("modelLangpair", this.state.modelReference.langpair);
      urlParams.set("modelModelName", this.state.modelReference.modelName);
    }
    urlParams.set("score", this.state.score);

    return urlParams;
  }

  /**
   * Updates the state in place, but does not update the history or UI.
   *
   * @param {Partial<State>} partialState
   */
  replaceState(partialState) {
    this.state = {
      ...this.state,
      ...partialState,
    };
  }

  /**
   * Updates the state, URL history, and view. Use this for an atomic update that
   * should be serialize dto the view.
   *
   * @param {Partial<State>} partialState
   */
  update(partialState) {
    this.replaceState(partialState);
    this.pushHistory();
    this.updateUI();
  }

  /**
   * Push the current state onto the history.
   */
  pushHistory() {
    const urlParams = this.stateToURLSearchParams();
    const url = new URL(window.location.href);
    const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
    history.pushState(urlStateManager, "", newLocation);
  }

  /**
   * This is a reactive function that updates the UI based on state changes. It should
   * be quick to run.
   */
  updateUI() {
    SearchFilter.onStateChange(this.state.searchString);
    ModelCardOverlay.onStateChange(this.state.modelReference);

    if (this.state.showModels) {
      elements.table.classList.add("show-models");
    } else {
      elements.table.classList.remove("show-models");
    }
    if (this.state.showCorpora) {
      elements.table.classList.add("show-corpora");
    } else {
      elements.table.classList.remove("show-corpora");
    }
    elements.searchFilter.value = this.state.searchString;

    const scoreRadio = elements.scores.querySelector(
      "#score-" + this.state.score
    );
    if (scoreRadio) {
      scoreRadio.setAttribute("checked", "");
    }
    document.body.dataset["score"] = this.state.score;
  }
}

/**
 * The state manager is statically initalized.
 */
const urlStateManager = new URLStateManager();
exposeAsGlobal("urlStateManager", urlStateManager);

/**
 * These are also statically initialized. After they are initially set, the view
 * must be update.
 */
/** @type {TrainingRun[] | null} */
let trainingRuns = null;

/**
 * The initialization function for the page.
 */
document.addEventListener("DOMContentLoaded", async () => {
  elements.table.querySelectorAll("th button").forEach((button, index) => {
    button.addEventListener("click", () => sortTable(index));
  });

  trainingRuns = await loadTrainingRuns();
  exposeAsGlobal("trainingRuns", trainingRuns);

  SearchFilter.setupHandlers();
  ModelCardOverlay.setupHandlers();
  setupScoreHandlers();

  urlStateManager.updateUI();

  sortByDate();

  elements.tableContainer.style.display = "block";
  elements.loading.style.display = "none";
});

/**
 * Find the index of the data key, sort the table by it.
 */
function sortByDate() {
  const tr = elements.thead.querySelector("tr");
  if (!tr) {
    throw new Error("Could not find the tr");
  }
  for (let index = 0; index < tr.children.length; index++) {
    if (tr.children[index].getAttribute("data-key") === "date") {
      sortTable(index, -1);
      break;
    }
  }
}

class SearchFilter {
  /**
   * Sets up the event handlers.
   */
  static setupHandlers() {
    elements.searchFilter.addEventListener("keyup", () => {
      SearchFilter.onStateChange(elements.searchFilter.value);
    });
    function pushSearchFilter() {
      urlStateManager.replaceState({
        searchString: elements.searchFilter.value,
      });
      urlStateManager.pushHistory();
    }
    elements.searchFilter.addEventListener("keyup", (event) => {
      if (event.key === "Enter") {
        pushSearchFilter();
      }
    });
    elements.searchFilter.addEventListener("blur", pushSearchFilter);
  }

  /**
   * Reactively handle state changes. This should be fast enough to be called many times
   * quickly.
   *
   * @param {string} search
   */
  static onStateChange(search) {
    search = search.trim();
    const trs = Array.from(elements.tbody.querySelectorAll("tr"));

    // Unhide everything.
    for (const tr of trs) {
      tr.style.display = "table-row";
    }

    if (!search.trim()) {
      // Nothing to search.
      return;
    }

    const { filters, terms } = parseSearchQuery(search);

    // Filter terms
    for (const tr of elements.tbody.querySelectorAll("tr")) {
      const rowText = tr.innerText.toLowerCase();
      for (const term of terms) {
        if (!rowText.includes(term)) {
          tr.style.display = "none";
          break;
        }
      }
    }

    for (const filter of filters) {
      // Find the table header
      if (!filter.key.match(/^[a-z-]+$/)) {
        continue;
      }
      const ths = elements.thead.querySelectorAll("th");

      let columnIndex = null;
      for (let i = 0; i < ths.length; i++) {
        if (ths[i].dataset.key === filter.key) {
          columnIndex = i;
          break;
        }
      }
      if (columnIndex === null) {
        continue;
      }

      for (const tr of trs) {
        const td = /** @type {HTMLElement} */ (tr.children[columnIndex]);
        const rowText = td.innerText.toLowerCase();
        if (filter.negated) {
          if (rowText.includes(filter.value)) {
            tr.style.display = "none";
          }
        } else if (!rowText.includes(filter.value)) {
          tr.style.display = "none";
        }
      }
    }
  }
}

/**
 * This is the overlay for the model view. It takes a model reference, looks up the
 * training run and model run. Note that the training runs are expected to already be
 * loaded in. {@link ModelCardOverlay.onStateChange} can be called very cheaply to update
 * the view anytime the state changes.
 *
 * @param {ModelReference | null} modelReference
 */
class ModelCardOverlay {
  /** @type {TrainingRun} */
  trainingRun;

  /** @type {ModelRun} */
  modelRun;

  /** @type {ModelReference} */
  modelReference;

  /**
   * @param {TrainingRun} trainingRun
   * @param {ModelRun} modelRun
   * @param {ModelReference} modelReference
   */
  constructor(trainingRun, modelRun, modelReference) {
    this.trainingRun = trainingRun;
    this.modelRun = modelRun;
    this.modelReference = modelReference;
  }

  static setupHandlers() {
    function hideOverlay() {
      urlStateManager.update({ modelReference: null });
    }
    elements.overlayCloseButton.addEventListener("click", hideOverlay);
    document.body.addEventListener("keyup", (event) => {
      if (event.key === "Escape") {
        hideOverlay();
      }
    });
    elements.overlay.addEventListener("click", (event) => {
      if (event.target === elements.overlay) {
        hideOverlay();
      }
    });
  }

  /**
   * Reactively handle a state change to the model reference. This function is fast
   * enough to be called reactively, and only initializes the code when it is needed.
   */
  static onStateChange(modelReference) {
    if (!modelReference) {
      document.body.classList.remove("overlay-show");
      elements.scrollContainer.removeAttribute("inert");
      return null;
    }
    if (!trainingRuns) {
      // The training runs aren't available yet.
      return null;
    }

    if (document.body.classList.contains("overlay-show")) {
      // The model is already being shown.
      return null;
    }

    const { name, langpair, modelName } = modelReference;

    const trainingRun = trainingRuns.find(
      (trainingRun) =>
        trainingRun.name === name && trainingRun.langpair == langpair
    );
    if (!trainingRun) {
      elements.error.style.display = "block";
      elements.error.innerText = `Could not find the model "${name}" (${langpair})`;
      return null;
    }

    const modelRun = trainingRun[modelName];
    if (!modelRun) {
      elements.error.style.display = "block";
      elements.error.innerText = `That model couldn't be found for "${name}" (${langpair})`;
      return null;
    }

    const overlay = new ModelCardOverlay(trainingRun, modelRun, modelReference);
    overlay.initialize();
  }

  initialize() {
    // Clear out any old view.
    elements.overlayContent.innerText = "";

    this.createHeaders();

    const detailsUL = create.ul({
      parent: elements.overlayContent,
    });

    this.initModelDetails(detailsUL);
    this.initArtifacts(detailsUL);
    this.initTrainingContinuation();
    this.initTrainingConfig();

    // Show the overlay.
    elements.scrollContainer.setAttribute("inert", "");
    document.body.classList.add("overlay-show");
  }

  createHeaders() {
    const { name, langpair, modelName } = this.modelReference;
    create.h1({
      children: `${name} (${langpair})`,
      parent: elements.overlayContent,
    });
    create.h2({
      children: modelNameToLabel(modelName),
      parent: elements.overlayContent,
    });
  }

  /**
   * @param {HTMLElement} parent
   */
  initModelDetails(parent) {
    const tbody = create.tbody();

    const { task_group_id: taskGroupId, task_id: taskId } = this.modelRun;
    const { langpair, name } = this.trainingRun;

    create.li({
      parent,
      children: [
        "Model Details",
        create.table({
          className: "details-table",
          children: [
            create.thead({
              children: [
                create.tr({
                  children: [
                    create.th({ children: "Label" }),
                    create.th({ children: "Details" }),
                  ],
                }),
              ],
            }),
            tbody,
          ],
        }),
      ],
    });

    /**
     * @param {string} label
     * @param {any} value
     */
    const createRow = (label, value) => {
      create.tr({
        parent: tbody,
        children: [
          create.td({ children: label }),
          create.td({ children: value ? value : "-" }),
        ],
      });
    };

    createRow("Date", this.modelRun.date.slice(0, "2025-01-01".length));
    createRow(
      "TaskGroup",
      create.a({
        children: this.modelRun.task_group_id,
        href: `https://firefox-ci-tc.services.mozilla.com/tasks/groups/${taskGroupId}`,
      })
    );
    createRow(
      "Task",
      create.a({
        children: this.modelRun.task_name,
        href: `https://firefox-ci-tc.services.mozilla.com/tasks/${taskId}`,
      })
    );

    // https://wandb.ai/moz-translations/cs-en/runs/teacher-1_ThgMJX?nw=nwuserepavlov
    // https://wandb.ai/moz-translations/cs-en/runs/teacher-1_LjL0bY
    const modelName = this.modelReference.modelName.replace("_", "-");
    const idPart = this.modelRun.task_group_id.slice(0, 6);

    createRow(
      "W&B",
      create.ul({
        children: [
          create.li({
            children: create.a({
              children: "Model Run",
              href: `https://wandb.ai/moz-translations/${langpair}/runs/${modelName}_${idPart}`,
            }),
          }),
          create.li({
            children: create.a({
              children: `Task Group ${taskGroupId}`,
              href: `https://wandb.ai/moz-translations/${langpair}/groups/${name}_${taskGroupId}/workspace`,
            }),
          }),
          create.li({
            children: [
              create.a({
                children: langpair,
                href: `https://wandb.ai/moz-translations/${langpair}/`,
              }),
              create.div({
                className: "wandb-filter",
                children: [
                  "Group by ",
                  create.span({ children: "Group" }),
                  ", filter by ",
                  create.span({ children: name }),
                ],
              }),
              create.div({
                className: "wandb-filter",
                children: [
                  `Or filter by regex: `,
                  create.span({
                    children: this.trainingRun.task_group_ids
                      .map((t) => t.slice(0, 6))
                      .join("|"),
                  }),
                ],
              }),
            ],
          }),
        ],
      })
    );
  }

  /**
   * @param {HTMLElement} parent
   */
  initArtifacts(parent) {
    create.li({
      parent,
      children: "Artifacts",
    });
    {
      const artifactsUL = create.ul({ parent });

      for (const url of this.modelRun.artifact_urls) {
        const urlParts = url.split("/");
        const fileName = urlParts[urlParts.length - 1];
        create.li({
          parent: artifactsUL,
          children: create.a({ children: fileName, href: url }),
        });
      }
    }
  }

  /**
   * @param {HTMLUListElement} detailsUL
   * @param {TrainingRun} trainingRun
   * @param {ModelRun} modelRun
   */
  createEvaluationTable(detailsUL, trainingRun, modelRun) {
    const tbody = create.tbody();

    create.li({
      parent: detailsUL,
      children: [
        "Flores Evaluation",
        create.table({
          className: "details-table",
          children: [
            create.thead({
              children: [
                create.tr({
                  children: [
                    create.th({ children: "Metric" }),
                    create.th({ children: "Value" }),
                  ],
                }),
              ],
            }),
            tbody,
          ],
        }),
      ],
    });

    /**
     * @param {string} metric
     * @param {string} value
     */
    const createMetricRow = (metric, value) => {
      create.tr({
        parent: tbody,
        children: [
          create.td({ children: metric }),
          create.td({ children: value ? value : "-" }),
        ],
      });
    };

    for (const metric of ["chrf", "bleu", "comet"]) {
      const value = modelRun.flores
        ? String(modelRun.flores[metric])
        : "Not available";
      createMetricRow(metric, value);
    }

    const googleFlores = getGoogleFloresCometScore(trainingRun, modelRun);
    if (googleFlores) {
      createMetricRow("comet (vs Google)", googleFlores.difference);
      createMetricRow("comet (Google)", googleFlores.score);
    } else {
      createMetricRow("Google Flores", "Not Available");
    }
  }

  /**
   * Creates the section that allows you to copy and paste the part of the config
   * for training continuation.
   */
  initTrainingContinuation() {
    const { name, langpair, modelName } = this.modelReference;

    // Only generate the header once, if it's required.
    let headerGenerated = false;

    /**
     * @param {string} text
     */
    const createTrainingHeader = (text) => {
      if (!headerGenerated) {
        headerGenerated = true;
        create.h2({
          parent: elements.overlayContent,
          children: "Training Continuation",
        });
        create.p({
          parent: elements.overlayContent,
          children: [
            "Re-use this model in another training run. See the ",
            create.a({
              children: "training continuation docs",
              href: "../docs/training/using-pretrained-models/",
            }),
            " for more information.",
          ],
        });
      }

      create.h4({
        parent: elements.overlayContent,
        children: text,
      });
    };

    switch (modelName) {
      case "backwards":
        createTrainingHeader("Back translation inference");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Use the ${langpair} model from the "${name}" training run for back translations.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-backwards:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: use",
            "      type: default",
            "",
          ].join("\n"),
        });
        break;
      case "teacher_1":
      case "teacher_2":
        createTrainingHeader("Teacher distillation inference");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Use the existing ${langpair} model from the "${name}" training run.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-teacher:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: use",
            "      type: default",
            "",
          ].join("\n"),
        });

        createTrainingHeader("Fine-tune the teacher");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Fine tune the ${langpair} model from the "${name}" training run.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-teacher:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: continue",
            "      type: default",
            "",
          ].join("\n"),
        });
        break;
      case "student":
        createTrainingHeader("Back translation inference");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Use the ${langpair} model from the "${name}" training run for back translations.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-backwards:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: use",
            "      type: default",
            "",
          ].join("\n"),
        });

        createTrainingHeader("Fine-tune the student");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Fine tune the ${langpair} model from the "${name}" training run.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-student:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: continue",
            "      type: default",
            "",
          ].join("\n"),
        });

        createTrainingHeader("Run evaluations and export");
        create.pre({
          parent: elements.overlayContent,
          children: [
            "experiment:",
            "  pretrained-models:",
            `    # Use the existing ${langpair} model from the "${name}" training run.`,
            "    # See: https://mozilla.github.io/translations/docs/training/using-pretrained-models/",
            "    train-student:",
            "      urls:",
            "        - " + this.modelRun.artifact_folder,
            "      mode: use",
            "      type: default",
            "",
          ].join("\n"),
        });

      case "student_finetuned":
      case "student_quantized":
      case "student_exported":
        // These don't support training continuation.
        break;
      default:
        // Ensure every type of model is supported.
        isNever(modelName);
    }
  }

  initTrainingConfig() {
    create.h2({
      parent: elements.overlayContent,
      children: "Training Config",
    });

    create.pre({
      parent: elements.overlayContent,
      children: jsonToYAML(this.modelRun.config),
    });
  }
}

/**
 * Fetches JSON data from a given URL.
 *
 * @param {string} url
 * @returns {Promise<Object>}
 */
async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.statusText}`);
  }
  return response.json();
}

/**
 * Fetches and displays the training runs list.
 * @returns {Promise<TrainingRun[]>}
 */
async function loadTrainingRuns() {
  const trainingRunListing = await fetchJSON(
    `${STORAGE_URL}/models/listing.json`
  );
  const promises = trainingRunListing.map(async (filename) => {
    /** @type {TrainingRun} */
    const trainingRun = await fetchJSON(`${STORAGE_URL}/${filename}`);
    try {
      const row = new TrainingRunRow(trainingRun);
      row.build();
    } catch (error) {
      elements.error.style.display = "block";
      elements.error.innerText = "Error building training run row.";
      console.error(error);
    }
    return trainingRun;
  });
  const results = await Promise.allSettled(promises);
  const rejected = results
    .filter(({ status }) => status == "rejected")
    // @ts-expect-error - Not sure why the allSettled disagrees.
    .map(({ reason }) => reason);
  const fulfilled = results
    .filter(({ status }) => status == "fulfilled")
    // @ts-expect-error - Not sure why the allSettled disagrees.
    .map(({ value }) => value);
  if (rejected.length) {
    console.error("Some fetches failed", rejected);
  }
  return fulfilled;
}

const displayName = new Intl.DisplayNames("en", { type: "language" });

/**
 * Everything needed to build a training run row.
 */
class TrainingRunRow {
  /** @type {TrainingRun} */
  trainingRun;

  /** @type {HTMLTableRowElement} */
  tr;

  /**
   * Construct the class with the required data.
   */
  constructor(trainingRun) {
    this.trainingRun = trainingRun;
    this.tr = create.tr({ parent: elements.tbody });
  }

  /**
   * Call all of the sub functions to build the parts of the row. These pieces are
   * broken out into separate methods to make the building process organized.
   */
  build() {
    this.createInitialColumns();
    this.createModelButtons();
    this.createCorporaLinks();
  }

  /**
   * Create the Name, Language, and Language Pair columns.
   */
  createInitialColumns() {
    const trainingRun = this.trainingRun;
    const languageTag =
      trainingRun.source_lang === "en"
        ? trainingRun.target_lang
        : trainingRun.source_lang;

    this.createFilterableButton("name", trainingRun.name);
    this.createFilterableButton(
      "language",
      displayName.of(languageTag) ?? languageTag
    );
    this.createFilterableButton("langpair", trainingRun.langpair);
    create.td({
      parent: this.tr,
      children: (trainingRun.date_started ?? "–").slice(0, "2025-01-01".length),
    });
  }

  /**
   * Creates a button that when clicked when apply the search filter
   *
   * @param {string} key
   * @param {string} value
   */
  createFilterableButton(key, value) {
    create.td({
      parent: this.tr,
      children: create.button({
        className: "button-text",
        children: value,
        onClick() {
          elements.searchFilter.value = value.includes(" ")
            ? `${key}:"${value}"`
            : `${key}:${value}`;

          urlStateManager.update({
            searchString: elements.searchFilter.value,
          });
        },
      }),
    });
  }

  /**
   * Create a single link to a model, that when clicked will open
   * @param {ModelName} modelName
   */
  createModelOverlayButton(modelName) {
    const div = document.createElement("div");
    const trainingRun = this.trainingRun;

    const modelRun = trainingRun[modelName];
    const googleFlores = getGoogleFloresCometScore(trainingRun, modelRun);
    const comet = modelRun?.flores?.comet;
    const bleu = modelRun?.flores?.bleu;

    /** @type {Partial<CSSStyleDeclaration>} */
    const style = {};
    let shippable = "Shippable";
    if (googleFlores && googleFlores.percentage < -5) {
      // Does not meet release criteria.
      style.background = "#ffa537";
      shippable = "Not shippable";
    }
    const title =
      `${shippable} - COMET ${comet?.toFixed(2)} ` +
      `vs Google Comet ${googleFlores?.score} ` +
      `(${googleFlores?.difference})`;

    create.td({
      parent: this.tr,
      className: "models-td",
      style,
      children: create.div({
        children: !trainingRun[modelName]
          ? "–"
          : create.button({
              parent: div,
              title,
              children: [
                create.span({
                  className: "score-vs-google",
                  children: googleFlores ? googleFlores.difference : "view",
                }),
                create.span({
                  className: "score-comet",
                  children: comet?.toFixed(2) || "view",
                }),
                create.span({
                  className: "score-bleu",
                  children: bleu?.toFixed(2) || "view",
                }),
              ],
              className: "button-text",
              onClick() {
                urlStateManager.update({
                  modelReference: {
                    name: trainingRun.name,
                    langpair: trainingRun.langpair,
                    modelName,
                  },
                });
              },
            }),
      }),
    });
  }

  /**
   * Create the show/hide button for the models, and the buttons that can open up
   * the model overlay.
   */
  createModelButtons() {
    // Create the button to show models.
    create.td({
      parent: this.tr,
      children: create.button({
        onClick() {
          urlStateManager.update({
            showModels: !urlStateManager.state.showModels,
          });
        },
        children: [
          create.span({
            className: "toggle-models-show",
            children: "Show",
          }),
          create.span({
            className: "toggle-models-hide",
            children: "Hide",
          }),
        ],
      }),
    });

    this.createModelOverlayButton("backwards");
    this.createModelOverlayButton("teacher_1");
    this.createModelOverlayButton("teacher_2");
    this.createModelOverlayButton("student");
    this.createModelOverlayButton("student_finetuned");
    this.createModelOverlayButton("student_quantized");
    this.createModelOverlayButton("student_exported");
  }

  /**
   * Create a link to the source and target parts of a corpus. If there is no corpus
   * then a "-" is added instead.
   *
   * @param {Corpus} [corpus]
   */
  createCorpusLink(corpus) {
    const div = document.createElement("div");
    const { source_lang, target_lang } = this.trainingRun;
    create.td({
      parent: this.tr,
      className: "corpus-td",
      children: create.div({
        children: corpus
          ? [
              create.a({
                children: source_lang,
                href: corpus.source_url,
                title: formatBytes(corpus.source_bytes),
                parent: div,
              }),
              create.a({
                children: target_lang,
                href: corpus.target_url,
                title: formatBytes(corpus.target_bytes),
                parent: div,
              }),
            ]
          : "–",
      }),
    });
  }

  /**
   * Build the show/hide button that shows the corpora links. And build out all of
   * the links to the various corpora.
   */
  createCorporaLinks() {
    const trainingRun = this.trainingRun;

    // Create the button to show corpora.
    create.td({
      parent: this.tr,
      children: create.button({
        onClick() {
          urlStateManager.update({
            showCorpora: !urlStateManager.state.showCorpora,
          });
        },
        children: [
          create.span({
            className: "toggle-corpora-show",
            children: "Show",
          }),

          create.span({
            className: "toggle-corpora-hide",
            children: "Hide",
          }),
        ],
      }),
    });

    this.createCorpusLink(trainingRun.parallel_corpus_aligned);
    this.createCorpusLink(trainingRun.backtranslations_corpus_aligned);
    this.createCorpusLink(trainingRun.distillation_corpus_aligned);
    this.createCorpusLink(trainingRun.parallel_corpus);
    this.createCorpusLink(trainingRun.backtranslations_corpus);
    this.createCorpusLink(trainingRun.distillation_corpus);

    create.td({
      parent: this.tr,
      children: create.button({
        children: "log",
        title: "View this run in the console.log",
        onClick() {
          alert("View this run in the console.log");
          console.log(trainingRun.name, trainingRun.langpair);
          console.log(trainingRun);
        },
      }),
    });
  }
}

let prevColumnIndex = -1;
let prevDirection = 1;

/**
 * Sort a table by a column. This quickly mutates the HTMLTableElement to reorder the
 * rows based on the TD's innerText property.
 *
 * @param {number} columnIndex
 */
function sortTable(columnIndex, defaultDirection = 1) {
  const rows = Array.from(elements.tbody.children);
  // Swap the direction on double clicks
  const direction =
    prevColumnIndex === columnIndex ? -prevDirection : defaultDirection;
  prevDirection = direction;
  prevColumnIndex = columnIndex;

  rows.sort((rowA, rowB) => {
    const valueA = rowA.querySelectorAll("td")[columnIndex].innerText;
    const valueB = rowB.querySelectorAll("td")[columnIndex].innerText;
    return String(valueA).localeCompare(String(valueB)) * direction;
  });

  // Re-appending puts this row at the bottom
  rows.forEach((row) => elements.tbody.appendChild(row));
}

/**
 * Refine a raw type to a proper {@link ModelName} or null.
 *
 * @param {string | null | undefined} modelName
 * @returns {ModelName | null}
 */
function toModelName(modelName) {
  switch (modelName) {
    case "backwards":
    case "teacher_1":
    case "teacher_2":
    case "student":
    case "student_finetuned":
    case "student_quantized":
    case "student_exported":
      return modelName;
    default:
      return null;
  }
}

/**
 * Get the human-readable label for a given {@link ModelName}.
 *
 * @param {ModelName} modelName
 */
function modelNameToLabel(modelName) {
  switch (modelName) {
    case "backwards":
      return "Backwards";
    case "teacher_1":
      return "Teacher 1";
    case "teacher_2":
      return "Teacher 2";
    case "student":
      return "Student";
    case "student_finetuned":
      return "Student Finetuned";
    case "student_quantized":
      return "Student Quantized";
    case "student_exported":
      return "Student Exported";
    default:
      isNever(modelName);
      throw new Error("Could not convert model name to label: " + modelName);
  }
}

/**
 * The Google comparison requires a bit of computation. This is done in this helper class
 * to make it consistent.
 *
 * @param {TrainingRun} trainingRun
 * @param {ModelRun} [modelRun]
 */
function getGoogleFloresCometScore(trainingRun, modelRun) {
  const googleFlores = trainingRun.comet_flores_comparison.google;
  if (!googleFlores || !modelRun?.flores?.comet) {
    return null;
  }
  const percentage = 100 * (1 - googleFlores / (modelRun?.flores.comet / 100));
  const sign = percentage >= 0 ? "+" : "";
  return {
    percentage,
    difference: `${sign}${percentage.toFixed(2)}`,
    score: `${(googleFlores * 100).toFixed(2)}`,
  };
}

function setupScoreHandlers() {
  for (const radio of elements.scores.querySelectorAll("input[type=radio]")) {
    radio.addEventListener("change", () => {
      urlStateManager.update({
        score: getCheckedScore(),
      });
    });
  }
}

function getCheckedScore() {
  let id = "";
  for (const input of elements.scores.querySelectorAll("input")) {
    if (input.checked) {
      id = input.id;
    }
  }
  return id.replace("score-", "") || "vs-google";
}
