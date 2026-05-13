import { getServer } from './utils.mjs';
import {
  fetchArtifact,
  fetchTaskGroup,
  getArtifactSize,
  listArtifacts,
} from './taskcluster.mjs';

/**
 * A local cached copy of taskcluster data.
 */
export class TaskclusterDB {
  /**
   * @param {IDBDatabase} db
   */
  constructor(db) {
    this.db = db;
  }

  static async open() {
    return new TaskclusterDB(await openDatabase());
  }

  /**
   * Adds a task group to the 'taskGroups' object store.
   * @param {TaskGroup} taskGroup - The task group to add to the database.
   * @returns {Promise<void>} A promise that resolves when the transaction is complete.
   */
  async addTaskGroup(taskGroup) {
    const db = await openDatabase();
    const transaction = db.transaction('taskGroups', 'readwrite');
    const store = transaction.objectStore('taskGroups');
    store.add(taskGroup);
    return awaitTransactionComplete(transaction);
  }

  /**
   * @param {ArtifactListing} artifactListing
   * @returns {Promise<void>}
   */
  async addArtifactListing(artifactListing) {
    const db = await openDatabase();
    const transaction = db.transaction('artifactListing', 'readwrite');
    const store = transaction.objectStore('artifactListing');
    store.add(artifactListing);
    return awaitTransactionComplete(transaction);
  }

  /**
   * Compute an artifact listing.
   *
   * @param {TaskAndStatus} taskAndStatus
   * @returns {Promise<ArtifactListing>}
   */
  async getArtifactListing({ task, status }) {
    const { taskId, runs } = status;

    if (!runs) {
      return {
        taskId,
        totalSize: 0,
        totalMonthBytes: 0,
        artifacts: [],
      };
    }
    let listingFromDB = await this.#getArtifactListingFromDB(taskId);
    if (listingFromDB) {
      console.log(`[db] cached artifact listing ${task.metadata.name}`);
      return listingFromDB;
    }
    const server = getServer();

    /** @type {ArtifactListing} */
    const artifactListing = {
      taskId,
      totalSize: 0,
      totalMonthBytes: 0,
      artifacts: [],
    };

    const monthsInStorage = getMonthsBetweenDates(task.created, task.expires);

    for (const { runId } of runs) {
      const { artifacts } = await listArtifacts(server, status.taskId, runId);
      for (const { name: path } of artifacts) {
        const size = await getArtifactSize(server, status.taskId, path);
        /** @type {null | number} */
        let monthBytes = null;
        if (size) {
          monthBytes = monthsInStorage * size;
          artifactListing.totalSize += size;
          artifactListing.totalMonthBytes += monthBytes;
        }

        artifactListing.artifacts.push({
          runId,
          path,
          size,
          monthBytes,
        });
      }
    }

    await this.addArtifactListing(artifactListing);
    return artifactListing;
  }

  /**
   * @param {string} taskGroupId
   * @returns {Promise<TaskGroup | null>}
   */
  async getTaskGroup(taskGroupId) {
    {
      const taskGroup = await this.#getTaskGroupFromDB(taskGroupId);
      if (taskGroup) {
        console.log('[db] Task group was in indexeddb', taskGroupId);
        return taskGroup;
      }
    }
    try {
      const taskGroup = await fetchTaskGroup(
        getServer(),
        taskGroupId,
        () => {},
      );
      console.log('[db] Saving taskgroup to indexeddb', taskGroup);
      await this.addTaskGroup(taskGroup);
      return taskGroup;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  /**
   * @param {string} taskId
   * @param {string} path
   * @returns {Promise<string | null>}
   */
  async getArtifactText(taskId, path) {
    {
      const entry = await this.#getArtifactTextFromDB(taskId, path);
      if (entry) {
        console.log('[db] Artifact was in indexeddb', taskId, path);
        return entry.text;
      }
    }
    try {
      const text = await fetchArtifact(
        getServer(),
        taskId,
        path,
        'text',
        false /* cache */,
      );
      console.log('[db] Saving artifact text to indexeddb', taskId, path);
      await this.#addArtifactText(taskId, path, text);
      return text;
    } catch (error) {
      console.error(error);
      return null;
    }
  }

  /**
   * @param {string} taskId
   * @param {string} path
   * @returns {Promise<any>}
   */
  async getArtifactJSON(taskId, path) {
    const text = await this.getArtifactText(taskId, path);
    if (text === null) {
      return null;
    }
    try {
      return JSON.parse(text);
    } catch (error) {
      console.error('[db] Failed to parse json', { text });
      return null;
    }
  }

  /**
   * @param {string} taskId
   * @param {string} path
   * @param {string} text
   * @returns {Promise<void>}
   */
  async #addArtifactText(taskId, path, text) {
    const db = await openDatabase();
    const transaction = db.transaction('artifactText', 'readwrite');
    const store = transaction.objectStore('artifactText');
    /** @type {ArtifactText} */
    const entry = { taskId, path, text };
    store.add(entry);
    return awaitTransactionComplete(transaction);
  }

  /**
   * @param {string} taskId
   * @param {string} artifactPath
   * @returns {Promise<ArtifactText | null>}
   */
  async #getArtifactTextFromDB(taskId, artifactPath) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('artifactText', 'readonly');
      const store = transaction.objectStore('artifactText');
      const request = store.get([taskId, artifactPath]);

      request.onsuccess = (event) => {
        resolve(getIDBRequest(event).result);
      };

      request.onerror = (event) => {
        reject(getIDBRequest(event).error);
      };
    });
  }

  /**
   * @param {string} taskGroupId
   * @returns {Promise<TaskGroup | null>}
   */
  async #getTaskGroupFromDB(taskGroupId) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('taskGroups', 'readonly');
      const store = transaction.objectStore('taskGroups');
      const request = store.get(taskGroupId);

      request.onsuccess = (event) => {
        resolve(getIDBRequest(event).result);
      };

      request.onerror = (event) => {
        reject(getIDBRequest(event).error);
      };
    });
  }

  /**
   * @param {string} taskId
   * @returns {Promise<ArtifactListing | null>}
   */
  async #getArtifactListingFromDB(taskId) {
    const db = await openDatabase();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction('artifactListing', 'readonly');
      const store = transaction.objectStore('artifactListing');
      const request = store.get(taskId);

      request.onsuccess = (event) => {
        resolve(getIDBRequest(event).result);
      };

      request.onerror = (event) => {
        reject(getIDBRequest(event).error);
      };
    });
  }

  /**
   * Deletes an IndexedDB database.
   * @returns {Promise<void>}
   */
  static delete() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.deleteDatabase('TaskclusterData');

      request.onsuccess = () => {
        console.log(`Database TaskclusterData deleted successfully`);
        resolve();
      };

      request.onerror = (event) => {
        console.error(
          `Error deleting database: ${getIDBOpenDBRequest(event).error}`,
        );
        reject(getIDBOpenDBRequest(event).error);
      };

      request.onblocked = () => {
        alert(`Deletion of database is blocked. Please close all connections.`);
      };
    });
  }
}

/**
 * Opens the IndexedDB database.
 * @returns {Promise<IDBDatabase>} A promise that resolves to the database instance.
 */
async function openDatabase() {
  // Rather than deal with migrations, just delete old databases, as they are only
  // used as caches.
  indexedDB.deleteDatabase('TaskclusterData');

  return new Promise((resolve, reject) => {
    const request = indexedDB.open('TaskclusterData2', 1);
    indexedDB.databases;

    request.onupgradeneeded = (event) => {
      if (!event.target) {
        return;
      }
      const db = getIDBOpenDBRequest(event).result;
      const tables = [
        { name: 'taskGroups', keyPath: 'taskGroupId' },
        { name: 'artifactListing', keyPath: 'taskId' },
        { name: 'artifactText', keyPath: ['taskId', 'path'] },
      ];
      for (const { name, keyPath } of tables) {
        const objectStore = db.createObjectStore(name, { keyPath });
        objectStore.createIndex(name, keyPath, { unique: true });
      }
    };

    request.onsuccess = (event) => {
      resolve(getIDBOpenDBRequest(event).result);
    };

    request.onerror = (event) => {
      reject(getIDBOpenDBRequest(event).error);
    };
  });
}

/**
 * @param {IDBTransaction} transaction
 * @returns {Promise<void>}
 */
function awaitTransactionComplete(transaction) {
  return new Promise((resolve) => {
    transaction.addEventListener('complete', () => void resolve());
  });
}

/**
 * @template T
 * @param {Event} event
 * @returns {IDBRequest<T>}
 */
function getIDBRequest(event) {
  const request = /** @type {IDBRequest<T> | null} */ (event.target);
  if (!request) {
    console.error(event);
    throw new Error('Expected to get the IDBRequest');
  }
  return request;
}

/**
 * @param {Event} event
 * @returns {IDBOpenDBRequest}
 */
function getIDBOpenDBRequest(event) {
  const request = /** @type {IDBOpenDBRequest | null} */ (event.target);
  if (!request) {
    console.error(event);
    throw new Error('Expected to get the IDBRequest');
  }
  return request;
}

/**
 * Computes the number of months between two date strings (ISO 8601 format).
 *
 * @param {string} a
 * @param {string} b
 * @returns {number}
 */
function getMonthsBetweenDates(a, b) {
  // Account for Gregorian year lengths because I'm a nerd.
  const daysInYear = 365 + 0.25 - 0.01 + 0.0025 - 0.00025;
  const daysInMonth = daysInYear / 12;
  const msInMonth = 1000 * 60 * 60 * 24 * daysInMonth;
  // @ts-ignore
  return (new Date(b) - new Date(a)) / msInMonth;
}
