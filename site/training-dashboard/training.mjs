/* vim: set tabstop=2 shiftwidth=2: */
import {
  exposeAsGlobal,
  getElement,
  replaceLocation,
  changeLocation,
} from './utils.mjs';
import { isTaskGroupIdValid } from './taskcluster.mjs';
import { TaskclusterDB } from './taskcluster-db.mjs';

const server = 'https://firefox-ci-tc.services.mozilla.com';

const elements = {
  dashboardName: /** @type {HTMLInputElement} */ (getElement('dashboardName')),
  taskGroup: /** @type {HTMLInputElement} */ (getElement('taskGroup')),
  loading: getElement('loading'),
  controls: getElement('controls'),
  table: /** @type {HTMLTableElement} */ (getElement('table')),
  tbody: getElement('tbody'),
  showAll: /** @type {HTMLInputElement} */ (getElement('showAll')),
  trainTaskGroupIds: getElement('trainTaskGroupIds'),
};

/** @type {Array<TaskGroup>} */
const taskGroups = [];

exposeAsGlobal('taskGroups', taskGroups);

document.addEventListener('DOMContentLoaded', () => {
  main().catch((error) => {
    console.error(error);
    getElement('error').style.display = 'block';
  });
});

async function main() {
  const db = await TaskclusterDB.open();
  elements.showAll.checked = getShowAll();
  elements.showAll.addEventListener('click', () => {
    const urlParams = new URLSearchParams(window.location.search);
    urlParams.set('showAll', elements.showAll.checked.toString());
    changeLocation(urlParams);
  });

  for (const [
    taskGroupIdIndex,
    { taskGroupId, name },
  ] of getTrainTaskGroups().entries()) {
    const { tr, createTD } = createTableRow(elements.trainTaskGroupIds);
    createTD('Train Task Group');
    {
      const a = document.createElement('a');
      a.innerText = taskGroupId;
      a.href = `${server}/tasks/groups/${taskGroupId}`;
      a.target = '_blank';
      createTD(a);
    }
    {
      const input = document.createElement('input');
      input.value = name;
      input.placeholder = 'Add a name';
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          const names = getTrainTaskGroups().map(({ name }) => name);
          names[taskGroupIdIndex] = input.value ?? '';

          const urlParams = new URLSearchParams(window.location.search);
          urlParams.set('taskGroupNames', JSON.stringify(names));
          replaceLocation(urlParams);
        }
      });
      createTD(input);
    }
    {
      const button = document.createElement('button');
      button.innerText = 'remove';
      button.addEventListener('click', () => {
        const ids = [];
        const names = [];
        for (const { taskGroupId: id, name } of getTrainTaskGroups()) {
          if (taskGroupId !== id) {
            ids.push(id);
            names.push(name);
          }
        }
        const urlParams = new URLSearchParams(window.location.search);
        urlParams.set(
          'taskGroupIds',
          ids.filter((id) => id !== taskGroupId).join(','),
        );
        urlParams.set('taskGroupNames', JSON.stringify(names));
        changeLocation(urlParams);
      });
      createTD(button);
    }
    elements.trainTaskGroupIds.appendChild(tr);
  }
  elements.taskGroup.addEventListener('keydown', (event) => {
    const taskGroupId =
      /** @type {HTMLInputElement } */ elements.taskGroup.value;
    if (event.key === 'Enter' && taskGroupId) {
      if (!isTaskGroupIdValid(taskGroupId)) {
        alert('The task group id was not valid');
        return;
      }
      const ids = getTrainTaskGroupIds();
      ids.push(taskGroupId);

      const urlParams = new URLSearchParams(window.location.search);
      urlParams.set('taskGroupIds', ids.join(','));
      changeLocation(urlParams);
    }
  });
  {
    const urlParams = new URLSearchParams(window.location.search);
    const title =
      urlParams.get('dashboardName') || 'Firefox Translations Training';
    elements.dashboardName.value = title;
    document.title = title;
    elements.dashboardName.addEventListener('keyup', () => {
      const urlParams = new URLSearchParams(window.location.search);
      urlParams.set('dashboardName', elements.dashboardName.value);
      replaceLocation(urlParams);
      document.title = elements.dashboardName.value;
    });
  }

  buildTable(db);
}

/**
 * @param {string} taskGroupId
 * @returns {Promise<TaskGroup>}
 */
async function fetchTaskGroup(taskGroupId) {
  const listUrl = `${server}/api/queue/v1/task-group/${taskGroupId}/list`;
  console.log('Fetching task group', listUrl);
  const response = await fetch(listUrl);
  if (!response.ok) {
    const error = await response.json();
    console.error();
    return Promise.reject(new Error(error));
  }
  return await response.json();
}

/**
 * @param {string} taskId
 * @returns {Promise<Task>}
 */
export async function fetchTask(taskId) {
  const taskUrl = `${server}/api/queue/v1/task/${taskId}`;
  const response = await fetch(taskUrl);
  if (!response.ok) {
    const error = await response.json();
    console.error();
    return Promise.reject(new Error(error));
  }
  return await response.json();
}

/**
 * For train actions, we need the task group ID that gets created as a result of the
 * train action.
 *
 * @param {string} taskId
 * @returns {Promise<null | string>}
 */
async function getDependentTaskGroupId(taskId) {
  const cacheKey = 'dependent-task-group-id-' + taskId;
  const cached = localStorage.getItem(cacheKey);
  if (cached) {
    return cached;
  }
  const { tasks } = await fetchDependents(taskId);
  const [firstTask] = tasks;
  if (!firstTask) {
    // No tasks have been scheduled yet.
    return null;
  }
  localStorage.setItem(cacheKey, firstTask.task.taskGroupId);
  return firstTask.task.taskGroupId;
}

/**
 * @param {string} taskId
 * @returns {Promise<TaskDependents>}
 */
async function fetchDependents(taskId) {
  const taskUrl = `${server}/api/queue/v1/task/${taskId}/dependents`;

  const response = await fetch(taskUrl);
  if (!response.ok) {
    const error = await response.json();
    console.error();
    return Promise.reject(new Error(error));
  }
  return await response.json();
}

/**
 * @param {HTMLElement} tbody
 */
function createTableRow(tbody) {
  const tr = document.createElement('tr');
  tbody.appendChild(tr);

  return {
    tr,
    /**
     * @param {string | Element} [textOrEl]
     * @returns {HTMLTableCellElement}
     */
    createTD(textOrEl = '') {
      const el = document.createElement('td');
      if (typeof textOrEl === 'string') {
        el.innerText = textOrEl;
      } else {
        el.appendChild(textOrEl);
      }
      tr.appendChild(el);
      return el;
    },
  };
}

/**
 * @param {TaskclusterDB} db
 */
function buildTable(db) {
  /**
   * Each task group here belongs to a language pair, and we need to assemble
   * various runs together into a single row on the table.
   *
   * @type {Map<string, Array<TaskAndStatus[]>>}
   */
  const taskGroupsByLangPair = new Map();

  const hiddenTaskGroups = getHiddenTaskGroups();
  const showAll = getShowAll();

  const fetchPromises = getTrainTaskGroupIds().map(async (trainTaskGroupId) => {
    // The actions all belong to a task group, fetch the information for this task group.
    const actionTaskGroup = await fetchTaskGroup(trainTaskGroupId);
    console.log('Action task group', actionTaskGroup);

    const trainTasks = actionTaskGroup.tasks.filter(
      ({ task }) => task.metadata.name === 'Action: Train',
    );

    console.log('trainTasks', trainTasks);

    // Go through all of the train actions.
    const promises = trainTasks.map(async (trainActionTask) => {
      const taskGroupId = await getDependentTaskGroupId(
        trainActionTask.status.taskId,
      );

      if (!taskGroupId) {
        // Create an empty table row, there is nothing else to do until tasks
        // are generated.
        const { createTD } = createTableRow(elements.tbody);
        if (showAll) {
          createTD(
            `${trainActionTask.status.taskId} ${trainActionTask.status.state}`,
          );
        }
      } else {
        // Build the full table row. Eventually these get deduplicated by language pair.
        const isHidden = hiddenTaskGroups.includes(taskGroupId);
        if (showAll || !isHidden) {
          const { tr, createTD } = createTableRow(elements.tbody);

          tr.dataset.taskGroupId = taskGroupId;
          await buildTableRow(
            db,
            createTD,
            taskGroupId,
            taskGroupsByLangPair,
            trainActionTask,
            isHidden,
          );
        }
      }
      // Sort the table by langpair after every row is built.
      scheduleTableRowSort();
    });

    // Wait until everything is settled. If a lookup fails this won't block the final
    // part of the script, but we still want to log the error.
    return Promise.all(promises.map((promise) => promise.catch(console.error)));
  });

  // Either hide or dim the older task groups depending on the value
  // of `showAll`.
  Promise.allSettled(fetchPromises).then(() => {
    for (let taskGroups of taskGroupsByLangPair.values()) {
      // Sort newest to oldest
      taskGroups.sort((aList, bList) => {
        const a = aList[0].task.created;
        const b = bList[0].task.created;
        return a < b ? 1 : a > b ? -1 : 0;
      });

      // Don't include hidden task groups.
      if (!showAll) {
        taskGroups = taskGroups.filter(
          (taskGroup) =>
            !hiddenTaskGroups.includes(taskGroup[0]?.task.taskGroupId),
        );
      }

      // Skip the first task group as it's the most recent, and shouldn't be dimmed.
      taskGroups = taskGroups.slice(1);

      for (const taskGroup of taskGroups) {
        const { taskGroupId } = taskGroup[0].task;
        /** @type {HTMLElement | null} */
        const tr = document.querySelector(
          `tr[data-task-group-id="${taskGroupId}"]`,
        );
        if (!tr) {
          throw new Error('Could not find tr for task group.');
        }
        if (showAll) {
          tr.classList.add('older-taskgroup');
        } else {
          tr.style.display = 'none';
        }
      }
    }

    // Expose this as a global after everything is fetched and processed.
    exposeAsGlobal('taskGroupsByLangPair', taskGroupsByLangPair);
  });

  elements.loading.style.display = 'none';
  elements.table.style.display = 'table';
}

/**
 * @typedef {object} ScoreDetails
 * @prop {string} langPair
 * @prop {string} langPairKey
 * @prop {number | null} score
 * @prop {Date} created
 * @prop {string} taskId
 * @prop {string | null} provider
 * @prop {string | null} dataset
 */

/**
 * The evaluation scores need to be found in tasks in the task groups. When they are
 * are found they need to be sorted to find the newest scores metrics. This is because
 * runs may be retriggered if the scores are too low.
 *
 * @type {Record<string, Array<ScoreDetails>>}
 */
const scores = {
  teacher1: [],
  teacher2: [],
  teacherensemble: [],
  student: [],
  finetunedstudent: [],
  studentquantized: [],
};

/**
 * As scores are pulled in they need to be accumulated and summarized into a row. Only
 * the newest score is used as models are retriggered for errors. If all of the rows
 * are shown then this update step is skipped.
 */
function updateScores() {
  if (getShowAll()) {
    // Do not update all of the scores if the scores are all shown.
    return;
  }
  for (const [key, scoresList] of Object.entries(scores)) {
    /**
     * Compute the latest scores for each language pair.
     *
     * @type {Map<string, ScoreDetails>}
     */
    const latestScores = new Map();
    for (const scoreDetails of scoresList) {
      const latestScoreDetails = latestScores.get(scoreDetails.langPairKey);
      if (
        !latestScoreDetails ||
        scoreDetails.created > latestScoreDetails.created
      ) {
        latestScores.set(scoreDetails.langPairKey, scoreDetails);
      }
    }

    // Update the COMET scores for a TD, or note that an eval score is still needed.
    for (const {
      langPair,
      langPairKey,
      score,
      provider,
      taskId,
      dataset,
    } of latestScores.values()) {
      for (const element of Array.from(
        document.querySelectorAll(`[data-${key}=${langPairKey}]`),
      )) {
        const td = /** @type {HTMLTableCellElement} */ (element);
        if (score === null) {
          // A null score means that a model was created, but there is no evaluation
          // result. This can happen depending on the target stage that is chosen.
          // Generally this indicates that an eval step needs to be triggered.
          while (td.lastChild) {
            td.lastChild.remove();
          }
          const a = document.createElement('a');
          a.innerText = 'Needs eval';
          a.href = `https://firefox-ci-tc.services.mozilla.com/tasks/${taskId}`;
          td.appendChild(a);
        } else {
          updateCometTD(td, langPair, score, provider, dataset, taskId);
        }
      }
    }
  }
}

/** @type {Promise<EvalResults>} */
let cometScores = {};
function getCometScores(langPair) {
  console.log("cometscores", cometScores);
  if (cometScores[langPair]) {
    return cometScores[langPair];
  }
  cometScores[langPair] = _getCometScores(langPair);
  return cometScores[langPair];
}

/**
 * @returns {Promise<EvalResults>}
 */
async function _getCometScores(langPair) {
  const url = `https://storage.googleapis.com/moz-fx-translations-data--303e-prod-translations-data/final-evals/${langPair}__flores200-plus__google__v2__latest__comet22.metrics.json`
  const response = await fetch(url,);

  return await response.json();
}

// /**
//  * @param {EvalResults} cometResults
//  * @param {string} langPair
//  * @return {number}
//  */
// function getAverageGoogleCometScore(cometResults, langPair) {
//   let googleScore = 0;
//   let googleScoreCount = 0;
//   if (cometResults[langPair]) {
//     for (const evals of Object.values(cometResults[langPair])) {
//       if (evals['google']) {
//         googleScore += evals['google'];
//         googleScoreCount++;
//       }
//     }
//     if (googleScoreCount) {
//       googleScore /= googleScoreCount;
//     }
//   }
//   return googleScore;
// }

/**
 * Update a TD with the relevant score information, and title text.
 *
 * @param {HTMLTableCellElement} td
 * @param {string} langPair
 * @param {number} score
 * @param {string | null} provider
 * @param {string | null} dataset
 * @param {string} taskId
 */
async function updateCometTD(td, langPair, score, provider, dataset, taskId) {
  const cometResults = await getCometScores(langPair);
  while (td.lastChild) {
    td.removeChild(td.lastChild);
  }

  let percentageDisplay;
  const googleScore = cometResults['score'];
  // const googleScore = getAverageGoogleCometScore(cometResults, langPair);
  const percentage = 100 * (1 - googleScore / score);
  const sign = percentage >= 0 ? '+' : '';
  if (provider === 'flores') {
    percentageDisplay = `${sign}${percentage.toFixed(2)}%`;

    let shippable = 'Shippable';
    td.style.color = '#fff';
    td.style.background = '#388e3c';
    if (percentage < -5) {
      // Does not meet release criteria.
      td.style.background = '#f44336';
      shippable = 'Not shippable';
    }

    td.title =
      `${shippable} - COMET ${score} ` +
      `vs Google Comet ${googleScore.toFixed(4)} ` +
      `(${percentageDisplay})`;
  } else if (provider && dataset) {
    td.title = `flores is not available, showing: ${provider}-${dataset}`;
  }

  {
    const a = document.createElement('a');
    a.href = `https://firefox-ci-tc.services.mozilla.com/tasks/${taskId}`;
    a.innerText = `${(score * 100).toFixed(2)}`;
    if (percentageDisplay) {
      a.style.color = '#fff';
    } else {
      a.style.color = '#000';
    }
    a.target = '_blank';
    td.appendChild(a);
  }

  if (percentageDisplay) {
    const span = document.createElement('span');
    span.innerText = percentageDisplay;
    span.style.color = '#000';
    td.appendChild(span);
  }
}

/**
 * Each task group gets its own row, which gets built here. At the end of all the fetches
 * the rows may be hidden. This function contains the most complicated logic. At the end
 * of it the rows are all sorted.
 *
 * @param {TaskclusterDB} db
 * @param {(text: string | Element) => HTMLTableCellElement} createTD
 * @param {string} taskGroupId
 * @param {Map<string, Array<TaskAndStatus[]>>} taskGroupsByLangPair
 * @param {TaskAndStatus} trainActionTask
 * @param {boolean} isHidden
 */
async function buildTableRow(
  db,
  createTD,
  taskGroupId,
  taskGroupsByLangPair,
  trainActionTask,
  isHidden,
) {
  const taskGroup = await fetchTaskGroup(taskGroupId);
  const experimentName = await getExperimentName(db, trainActionTask);
  const tasks = taskGroup.tasks;
  const taskGroupUrl = `${server}/tasks/groups/${taskGroupId}`;

  {
    // Build the task group ID link
    const div = document.createElement('div');
    div.className = 'taskGroupCell';

    const taskGroupLink = document.createElement('a');
    const showHideButton = document.createElement('button');

    const copyButton = document.createElement('button');
    copyButton.title = 'Copy the TaskGroup ID';
    copyButton.className = 'copyButton';
    copyButton.addEventListener('click', () => {
      navigator.clipboard.writeText(taskGroupId).catch((err) => {
        console.error(err);
        alert('Failed to copy taskGroupId');
      });
    });

    taskGroupLink.innerText = taskGroupId;
    taskGroupLink.href = taskGroupUrl;
    taskGroupLink.target = '_blank';

    showHideButton.innerText = isHidden ? 'Show' : 'Hide';
    showHideButton.className = 'showHideButton';
    showHideButton.addEventListener(
      'click',
      toggleHiddenHandler(taskGroupId, isHidden),
    );

    div.appendChild(copyButton);
    div.appendChild(taskGroupLink);
    div.appendChild(showHideButton);

    createTD(div);
  }

  {
    const div = document.createElement('div');
    div.className = 'taskGroupCell';

    const { env } = trainActionTask.task.payload;
    const a = document.createElement('a');
    const ref = env['TRANSLATIONS_BASE_REF'];
    const hash = env['TRANSLATIONS_BASE_REV'];
    a.innerText = env['TRANSLATIONS_BASE_REF'];
    a.href = `https://github.com/mozilla/translations/commits/${ref}`;

    const copyButton = document.createElement('button');
    copyButton.title = `Copy the revision hash ${hash}`;
    copyButton.className = 'copyButton';
    copyButton.addEventListener('click', () => {
      navigator.clipboard.writeText(hash).catch((err) => {
        console.error(err);
        alert('Failed to copy hash');
      });
    });

    div.appendChild(copyButton);
    div.appendChild(new Text(' '));
    div.appendChild(a);

    createTD(div);
  }

  // Attempt to find a langpair
  const langPair = await getLangPair(db, trainActionTask);

  const langPairKey = `${experimentName}-${langPair}`;
  {
    // Keep track of this list.
    let list = taskGroupsByLangPair.get(langPairKey);
    if (!list) {
      list = [];
      taskGroupsByLangPair.set(langPairKey, list);
    }
    list.push(tasks);
  }

  {
    const button = document.createElement('button');
    button.innerHTML = 'config';
    button.addEventListener(
      'click',
      copyConfigHandler(db, button, trainActionTask),
    );
    const a = document.createElement('a');
    a.innerText = langPair;
    a.href = `https://wandb.ai/moz-translations/${langPair}/workspace`;
    a.target = '_blank';
    const td = createTD(experimentName + ' ');
    td.appendChild(a);

    // Add a hidden date for sorting.
    const hiddenDate = document.createElement('span');
    hiddenDate.className = 'hiddenDate';
    hiddenDate.innerText = tasks[0].task.created;

    td.appendChild(hiddenDate);
    td.appendChild(document.createTextNode(' '));
    td.appendChild(button);
    td.appendChild(new Text(' '));
  }

  // The "[a-z]{2,3}-[a-z]{2,3}" part of the regexes below all match the language
  // pair, so for instance "en-ca". It supports langtags of length 2-3.

  /** @type {Array<{ name: string, evalMatch: RegExp[], trainMatch: RegExp | null}>} */
  const evals = [
    // Patterns before the rename:
    {
      name: 'teacher1',
      evalMatch: [
        /^evaluate-teacher-flores-devtest-[a-z]{2,3}-[a-z]{2,3}-1$/,
        /^evaluate-teacher-.*-[a-z]{2,3}-[a-z]{2,3}-1$/,
      ],
      // Old name: train-teacher-en-hu-1
      // New name: train-teacher-model-en-hu-1
      trainMatch: /^train-teacher-(model-)?[a-z]{2,3}-[a-z]{2,3}-1$/,
    },
    {
      name: 'teacher2',
      evalMatch: [
        /^evaluate-teacher-flores-devtest-[a-z]{2,3}-[a-z]{2,3}-2/,
        /^evaluate-teacher-.*-[a-z]{2,3}-[a-z]{2,3}-2/,
      ],
      trainMatch: /^train-teacher-(model-)?[a-z]{2,3}-[a-z]{2,3}-2$/,
    },
    {
      name: 'teacherensemble',
      evalMatch: [
        /^evaluate-teacher-ensemble-flores-devtest-[a-z]{2,3}-[a-z]{2,3}$/,
        /^evaluate-teacher-ensemble-.*-[a-z]{2,3}-[a-z]{2,3}$/,
      ],
      trainMatch: null,
    },
    {
      // Old name: train-student-en-hu
      // New name: distillation-student-model-train-en-hu
      name: 'student',
      evalMatch: [
        /^evaluate-student-flores-devtest-[a-z]{2,3}-[a-z]{2,3}$/,
        /^evaluate-student-.*-[a-z]{2,3}-[a-z]{2,3}$/,
      ],
      trainMatch:
        /^(distillation-student-model-train|train-student)-[a-z]{2,3}-[a-z]{2,3}$/,
    },
    {
      // Old name: finetune-student-en-hu
      // New name: distillation-student-model-finetune-en-hu
      name: 'finetunedstudent',
      evalMatch: [
        /^evaluate-finetuned-student-flores-devtest-[a-z]{2,3}-[a-z]{2,3}$/,
        /^evaluate-finetuned-student-.*-[a-z]{2,3}-[a-z]{2,3}$/,
      ],
      trainMatch:
        /^(distillation-student-model-finetune|finetune-student)-[a-z]{2,3}-[a-z]{2,3}$/,
    },
    {
      // Old name: quantize-en-hu
      // New name: distillation-student-model-quantize-en-hu
      name: 'studentquantized',
      evalMatch: [
        /^evaluate-quantized-flores-devtest-[a-z]{2,3}-[a-z]{2,3}$/,
        /^evaluate-quantized-.*-[a-z]{2,3}-[a-z]{2,3}$/,
      ],
      trainMatch:
        /^(distillation-student-model-quantize|quantize)-[a-z]{2,3}-[a-z]{2,3}$/,
    },
  ];

  for (const { name, evalMatch, trainMatch } of evals) {
    const scoreList = scores[name];
    /** @type {TaskAndStatus | undefined} */
    let evalTask;
    for (const regex of evalMatch) {
      if (evalTask) {
        break;
      }
      for (const task of tasks) {
        if (task.task.metadata.name.match(regex)) {
          if (task.status.state === 'completed') {
            evalTask = task;
          }
          break;
        }
      }
    }
    let trainTask;
    if (trainMatch) {
      trainTask = tasks.find(
        (t) =>
          t.task.metadata.name.match(trainMatch) &&
          t.status.state === 'completed',
      );
    }

    let td = createTD('');
    if (evalTask) {
      // If there is an eval teacher, pull its score, and update all of the other TDs,
      // as the task may have failed or be outdated, but its score is still valid.
      td.innerText = '';
      const { taskId } = evalTask.status;

      // Simplify the name for the regex.
      // From: "evaluate-teacher-mtdata-Neulab-tedtalks_test-1-eng-nor-no-en-1"
      //   To:                  "mtdata-Neulab-tedtalks_test-1-eng-nor-no-en-1"
      const name = evalTask.task.metadata.name
        .replace('evaluate-teacher-', '')
        .replace('evaluate-teacher-ensemble-', '')
        .replace('evaluate-student-', '')
        .replace('evaluate-finetuned-student-', '')
        .replace('evaluate-quantized-', '');

      // Find the dataset name from the task name.
      const match = name.match(
        // From: "mtdata-Neulab-tedtalks_test-1-eng-nor-no-en-1"
        // Match:       "Neulab-tedtalks_test-1-eng-nor"
        /^([a-zA-Z]+)-(.*)-[a-z]{2,3}-[a-z]{2,3}(-\d)?$/,
        //([a-zA-Z]+)                                    Provider, e.g. "mtdata", "flores"
        //            (.*)                               Datset, e.g. "Neulab-tedtalks_test-1", or "devtest"
        //                 [a-z]{2,3}                    Language tag
        //                            [a-z]{2,3}         Language tag
        //                                      (-\d)?   An optional number for ensembles.
      );

      if (match) {
        // "flores", "mtdata"
        const provider = match[1];
        // "devtest", "Neulab-tedtalks_test-1-eng-nor"
        const dataset = match[2];
        db.getArtifactJSON(taskId, `public/build/${dataset}.metrics.json`).then(
          (metrics) => {
            const score = metrics?.comet?.score;
            if (score) {
              scoreList.push({
                langPair,
                langPairKey,
                score,
                created: new Date(evalTask.task.created),
                taskId,
                provider,
                dataset,
              });
              updateCometTD(td, langPair, score, provider, dataset, taskId);
              updateScores();
            } else {
              console.error('Could not find the COMET score', {
                metrics,
                evalTask,
              });
            }
          },
        );
      } else {
        console.error(
          'Could not find the dataset name:',
          evalTask.task.metadata.name,
        );
      }
    } else if (trainTask) {
      scoreList.push({
        langPair,
        langPairKey,
        score: null,
        created: new Date(trainTask.task.created),
        taskId: trainTask.status.taskId,
        provider: null,
        dataset: null,
      });
    }

    td.setAttribute(`data-${name}`, langPairKey);
  }

  console.log(langPair, tasks);

  /** @type {Record<TaskState, number>} */
  const stateCounts = {
    completed: 0,
    running: 0,
    failed: 0,
    exception: 0,
    pending: 0,
    unscheduled: 0,
  };

  /**
   * @param {string[]} keys
   * @param {any} defaultValue
   * @returns {Record<string, any>}
   */
  function makeObj(keys, defaultValue) {
    /** @type {Record<string, any>} */
    const obj = {};

    for (const key of keys) {
      obj[key] = defaultValue;
    }
    return obj;
  }

  /**
   * Training is a complicated graph, but attempt to order the results here.
   */
  const orderedSteps = [
    'Action: Cancel All',
    'docker',
    'toolchain',
    'fetch',
    'continuation',

    // ---------------------------
    // Download and clean datasets
    'dataset',
    // datasets parallel
    'corpus-clean-parallel',
    'corpus-clean-parallel-fetch-bicleaner-model',
    'corpus-clean-parallel-bicleaner-ai',
    'corpus-merge-parallel',
    'corpus-analyze-parallel',
    'corpus-align-parallel',
    'build-vocab',
    // mono and devset tasks are faster
    'corpus-clean-mono',
    'corpus-merge-mono',
    'corpus-analyze-mono',
    'corpus-merge-devset',

    // ---------------------------
    // Synthesize Backtranslations
    'backtranslations-train-backwards-model',
    'evaluate-backward',
    'backtranslations-mono-trg-chunk',
    'backtranslations-mono-trg-translate',
    'backtranslations-mono-trg-dechunk',
    'corpus-align-backtranslation',

    // ---------------------------
    // Train teacher
    'train-teacher-model',
    'evaluate-teacher',

    // ----------------------------
    // Synthesize Distillation Data

    // Distill the parallel data
    'distillation-parallel-src-chunk',
    'distillation-parallel-src-translate',
    'distillation-parallel-src-extract-best',
    'distillation-parallel-src-dechunk-translation',

    // Distill the monolingual data
    'distillation-mono-src-chunk',
    'distillation-mono-src-translate',
    'distillation-mono-src-dechunk-translation',

    // Build the final distillation corpus
    'corpus-merge-distillation',
    'distillation-parallel-src-translations-score',
    'distillation-corpus-final-filtering',
    'corpus-align-distillation',

    // ----------------------------
    // Synthesize Distillation Data
    'distillation-corpus-build-shortlist',
    'distillation-student-model-train',
    'evaluate-student',
    'distillation-student-model-finetune',
    'evaluate-finetuned-student',
    'distillation-student-model-quantize',
    'evaluate-quantized',

    'export',
  ];

  /** @type {Record<string, "not-started" | "running" | "completed">} */
  const stepsCompleted = makeObj(orderedSteps, 'not-started');

  /** @type {Record<string, boolean>} */
  const stepsFailed = makeObj(orderedSteps, false);

  /** @type {Record<string, boolean>} */
  const stepsException = makeObj(orderedSteps, false);

  for (const { status, task } of tasks) {
    // Compute the status counts
    const count = stateCounts[status.state];
    stateCounts[status.state] = count + 1;

    for (const taskNamePrefix of Object.keys(stepsCompleted)) {
      if (task.metadata.name.startsWith(taskNamePrefix)) {
        // Compute if a step was completed.
        if (status.state === 'completed') {
          if (stepsCompleted[taskNamePrefix] === 'not-started') {
            stepsCompleted[taskNamePrefix] = 'completed';
          }
        } else if (status.state === 'running') {
          stepsCompleted[taskNamePrefix] = 'running';
        }

        if (status.state === 'failed') {
          stepsFailed[taskNamePrefix] = true;
        }
        if (status.state === 'exception') {
          stepsException[taskNamePrefix] = true;
        }
      }
    }
  }

  /**
   * @param {TaskState} state
   * @param {string} [color]
   * @param {string} [stage]
   */
  const addStateCount = (state, color, stage) => {
    if (!stateCounts[state]) {
      createTD('');
      return;
    }
    const a = document.createElement('a');
    a.innerText = String(stateCounts[state] ?? 0);
    a.target = '_blank';
    a.href = taskGroupUrl;
    const matchingTasks = tasks.filter((task) => task.status.state === state);
    if (matchingTasks.length === 1) {
      a.href = `${server}/tasks/${matchingTasks[0].status.taskId}`;
    }
    const el = createTD(a);
    if (color && stateCounts[state]) {
      el.style.background = color;
      a.style.color = '#fff';
    }
    if (stage) {
      const span = document.createElement('span');
      span.innerText = ' ' + stage;
      el.appendChild(span);
    }
  };

  let completed = '';
  let running = '';
  let failed = '';
  let exception = '';
  for (const [step, state] of Object.entries(stepsCompleted)) {
    if (state === 'completed') {
      completed = step;
    }
    if (state === 'running') {
      running = step;
    }
  }
  for (const [step, didFail] of Object.entries(stepsFailed).reverse()) {
    if (didFail) {
      failed = step;
    }
  }
  for (const [step, isException] of Object.entries(stepsException).reverse()) {
    if (isException) {
      exception = step;
    }
  }

  addStateCount('completed', undefined, completed);
  addStateCount('running', '#1976d2', running);
  addStateCount('failed', '#f44336', failed);
  addStateCount('exception', '#ffa000', exception);
  addStateCount('pending');
  addStateCount('unscheduled');

  updateScores();
}

/**
 * Hide or show a task group after a click.
 *
 * @param {string} taskGroupId
 * @param {boolean} isHidden
 * @returns {() => void}
 */
function toggleHiddenHandler(taskGroupId, isHidden) {
  return () => {
    let hidden = getHiddenTaskGroups();
    if (isHidden) {
      hidden = hidden.filter((id) => id !== taskGroupId);
    } else {
      hidden.push(taskGroupId);
    }
    const urlParams = new URLSearchParams(window.location.search);
    urlParams.set('hidden', hidden.join(','));
    changeLocation(urlParams);
  };
}

function getHiddenTaskGroups() {
  const urlParams = new URLSearchParams(window.location.search);
  const hiddenParam = urlParams.get('hidden');
  // e.g. "PuI6mYZPTUqAfyZMTgeUng,S5E71GihQM6Te_KdrUmATw"
  if (!hiddenParam) {
    return [];
  }
  return hiddenParam.split(',');
}

/**
 * Fetch the yml config for a training action.
 *
 * @param {TaskclusterDB} db
 * @param {TaskAndStatus} trainActionTask
 * @returns {Promise<string>}
 */
async function getConfigText(db, trainActionTask) {
  const configText = await db.getArtifactText(
    trainActionTask.status.taskId,
    'public/parameters.yml',
  );
  if (!configText) {
    return '';
  }

  // Extract the config
  const parts = configText.split('\ntraining_config:\n');

  // Collect all the lines of the same indent level
  let finalConfig = '';
  for (const line of parts[1].split('\n')) {
    if (line.startsWith('  ')) {
      finalConfig += line.slice(2) + '\n';
    } else {
      break;
    }
  }
  return finalConfig;
}

/**
 * Fetch the yml config for a training action.
 *
 * @param {TaskclusterDB} db
 * @param {TaskAndStatus} trainActionTask
 * @returns {Promise<string>}
 */
async function getExperimentName(db, trainActionTask) {
  const text = await getConfigText(db, trainActionTask);
  const experimentText = text.split('\nexperiment:\n')[1] ?? '';
  const nameText = experimentText.split('name:')[1] ?? '';
  return (nameText.split('\n')[0] ?? '').trim();
}

/**
 * Fetch the yml config for a training action.
 *
 * @param {TaskclusterDB} db
 * @param {TaskAndStatus} trainActionTask
 * @returns {Promise<string>}
 */
async function getLangPair(db, trainActionTask) {
  const text = await getConfigText(db, trainActionTask);
  const experimentText = text.split('\nexperiment:\n')[1] ?? '';
  const srcText = experimentText.split('  src:')[1] ?? '';
  let src = srcText.split('\n')[0] ?? '';
  const trgText = experimentText.split('  trg:')[1] ?? '';
  let trg = trgText.split('\n')[0] ?? '';

  // Remove any quotes form the word.
  src = src.trim().replaceAll("'", '').replaceAll('"', '');
  trg = trg.trim().replaceAll("'", '').replaceAll('"', '');

  return `${src}-${trg}`;
}

/**
 * @param {TaskclusterDB} db
 * @param {HTMLButtonElement} button
 * @param {TaskAndStatus} trainActionTask
 */
function copyConfigHandler(db, button, trainActionTask) {
  return async () => {
    try {
      button.innerText = 'downloading...';
      button.disabled = true;

      await navigator.clipboard.writeText(
        await getConfigText(db, trainActionTask),
      );

      button.innerText = 'config copied';
      button.disabled = false;
    } catch (error) {
      alert('Failed to get the training config');
    }
  };
}

function getTrainTaskGroupIds() {
  const urlParams = new URLSearchParams(window.location.search);
  // Extract the taskGroupId parameter
  const taskGroupIdParam = urlParams.get('taskGroupIds');

  // "PuI6mYZPTUqAfyZMTgeUng", "S5E71GihQM6Te_KdrUmATw"

  if (!taskGroupIdParam) {
    return [];
  }

  // Parse the taskGroupId values into an array
  const taskGroupIds = taskGroupIdParam.split(',');
  return taskGroupIds;
}

function getEmptyTrainTaskGroups() {
  return getTrainTaskGroupIds().map((taskGroupId) => ({
    taskGroupId,
    name: '',
  }));
}

/**
 * @returns {Array<{ taskGroupId: string, name: string }>}
 */
function getTrainTaskGroups() {
  const urlParams = new URLSearchParams(window.location.search);
  // Extract the taskGroupId parameter
  const namesString = urlParams.get('taskGroupNames');
  const ids = getTrainTaskGroupIds();

  if (!namesString) {
    return getEmptyTrainTaskGroups();
  }

  // Parse the taskGroupId values into an array
  let names;
  try {
    names = JSON.parse(namesString);
  } catch {}

  if (!Array.isArray(names)) {
    console.error('Bad taskGroupNames');
    return getEmptyTrainTaskGroups();
  }
  const namesFinal = [];
  for (let i = 0; i < ids.length; i++) {
    const name = typeof names[i] === 'string' ? names[i] : '';
    namesFinal.push({ taskGroupId: ids[i], name });
  }
  return namesFinal;
}

let isScheduled = false;

/**
 * Schedule the table sorts as they are quite slow.
 */
function scheduleTableRowSort() {
  if (!isScheduled) {
    isScheduled = true;
    requestAnimationFrame(() => {
      sortTable(elements.table, /* Lang pair column index */ 2);
      isScheduled = false;
    });
  }
}

/**
 * Kind of a hacky function to sort a table based on a column and index.
 * @param {HTMLTableElement} table
 * @param {number} columnIndex
 */
function sortTable(table, columnIndex, dir = 'asc') {
  let switchCount = 0;
  let shouldSwitch = false;
  let switching = true;
  while (switching) {
    switching = false;
    // @ts-ignore
    const rows = table.rows;

    let i;
    for (i = 1; i < rows.length - 1; i++) {
      shouldSwitch = false;

      const x = rows[i].querySelectorAll('td')[columnIndex]?.textContent ?? '';

      const y =
        rows[i + 1].querySelectorAll('td')[columnIndex]?.textContent ?? '';

      if (dir == 'asc') {
        if (x > y) {
          shouldSwitch = true;
          break;
        }
      } else if (dir == 'desc') {
        if (x < y) {
          shouldSwitch = true;
          break;
        }
      }
    }

    if (shouldSwitch) {
      rows[i].parentNode?.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      switchCount++;
    } else {
      if (switchCount == 0 && dir == 'asc') {
        dir = 'desc';
        switching = true;
      }
    }
  }
}

/**
 * Should the task chunks be merged?
 *
 * @returns {boolean}
 */
function getShowAll() {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get('showAll') === 'true';
}
