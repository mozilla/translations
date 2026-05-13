/**
 * @param {any} any
 * @returns {any}
 */
export function asAny(any) {
  return any;
}

/**
 * Ensure some T exists when the type systems knows it can be null or undefined.
 *
 * @template T
 * @param {T | null | undefined} item
 * @param {string} [message]
 * @returns {T}
 */
export function ensureExists(item, message = 'an item') {
  if (item === null) {
    throw new Error(message || 'Expected ${name} to exist, and it was null.');
  }
  if (item === undefined) {
    throw new Error(
      message || 'Expected ${name} to exist, and it was undefined.',
    );
  }
  return item;
}

const ENCODING_DIGITS =
  '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ._';
const LEADING_ZERO_DIGIT = ENCODING_DIGITS[0b100000];

/**
 * @param {Set<number>} numbers
 * @returns {string}
 */
export function encodeUintSetForUrlComponent(numbers) {
  // A set has no order. Convert it to an array and then sort the array,
  // so that consecutive numbers can be detected by encodeUintArrayForUrlComponent.
  const array = Array.from(numbers);
  array.sort((a, b) => a - b);
  return encodeUintArrayForUrlComponent(array);
}

/**
 * @param {number[]} numbers
 * @returns {string}
 */
export function encodeUintArrayForUrlComponent(numbers) {
  let result = '';
  for (let i = 0; i < numbers.length; i++) {
    const skipCount = countSkippableConsecutiveNumbersAt(numbers, i);
    if (skipCount === 0) {
      result += encodeUint(numbers[i]);
      continue;
    }

    i += skipCount;

    // We use the "leading zero digit" as the range marker.
    result += LEADING_ZERO_DIGIT;
    result += encodeUint(numbers[i]);
  }
  return result;
}

/**
 * @param {number[]} numbers
 * @param {number} start
 * @returns {number}
 */
function countSkippableConsecutiveNumbersAt(numbers, start) {
  if (start < 1 || start + 1 >= numbers.length) {
    return 0;
  }
  const previous = numbers[start - 1];
  const current = numbers[start];
  const next = numbers[start + 1];

  let skipCount = 0;
  if (current === previous + 1 && next === current + 1) {
    // Found increasing consecutive range.
    skipCount = 1;
    while (
      start + skipCount + 1 < numbers.length &&
      numbers[start + skipCount + 1] === current + skipCount + 1
    ) {
      skipCount++;
    }
  } else if (current === previous - 1 && next === current - 1) {
    // Found decreasing consecutive range.
    skipCount = 1;
    while (
      start + skipCount + 1 < numbers.length &&
      numbers[start + skipCount + 1] === current - skipCount - 1
    ) {
      skipCount++;
    }
  }
  return skipCount;
}

/**
 * @param {number} value
 * @returns {string}
 */
function encodeUint(value) {
  // Build the string digit by digit, back to front. The last digit has the
  // continuation bit set to 0, the other digits have it set to 1.
  // No "leading zero" digits are emitted, so that smaller numbers use fewer
  // digits, and so that "leading zero" digits can have special meaning.
  let x = value;
  let r = ENCODING_DIGITS[x & 0b11111];
  x >>= 5;
  while (x !== 0) {
    r = ENCODING_DIGITS[0b100000 + (x & 0b11111)] + r;
    x >>= 5;
  }
  return r;
}

export function getServer() {
  const urlParams = new URLSearchParams(window.location.search);
  const text = urlParams.get('server');
  if (!text) {
    return 'https://firefox-ci-tc.services.mozilla.com';
  }
  try {
    const url = new URL(text);
    return url.toString();
  } catch (error) {
    return 'https://firefox-ci-tc.services.mozilla.com';
  }
}

/**
 * @param {string} key
 * @param {any} value
 */
export function exposeAsGlobal(key, value) {
  console.log(key, value);
  asAny(window)[key] = value;
}

/**
 * Gets an element and throws if it doesn't exists.
 *
 * @param {string} id
 * @returns {HTMLElement}
 */
export function getElement(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error('Could not find element by id: ' + id);
  }
  return el;
}

/**
 * @param {HTMLElement} tbody
 * @param {Element?} [insertBefore]
 */
export function createTableRow(tbody, insertBefore) {
  const tr = document.createElement('tr');
  tbody.insertBefore(tr, insertBefore ?? null);

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
 * @param {TaskAndStatus[]} tasks
 * @returns {string}
 */
export function getLangPair(tasks) {
  for (const { task } of tasks) {
    if (task.metadata.name.match(/-src-[a-z]{2,3}$/)) {
      // Monolingual task.
      continue;
    }
    const match = task.metadata.name.match(/-([a-z]{2,3}-[a-z]{2,3})$/);
    if (match) {
      return match[1];
    }
  }
  return '';
}

/**
 * Formats a number of bytes into a human-readable string.
 *
 * @param {number} bytes
 * @param {number} [decimals]
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 2) {
  if (bytes === 0) return '0 B';

  const k = 1000;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * @template A
 * @template B
 * @param {Array<A>} a
 * @param {Array<B>} b
 * @returns {Generator<[A, B]>}
 */
export function* zip(a, b) {
  const length = Math.max(a.length, b.length);
  for (let i = 0; i < length; i++) {
    yield [a[i], b[i]];
  }
}

/**
 * @param {URLSearchParams} urlParams
 */
export function replaceLocation(urlParams) {
  const url = new URL(window.location.href);
  const newLocation = `${url.origin}${url.pathname}?${urlParams}`;
  history.replaceState(null, '', newLocation);
}

/**
 * @param {URLSearchParams} urlParams
 */
export function changeLocation(urlParams) {
  const url = new URL(window.location.href);
  const newLocation = `${url.origin}${url.pathname}?${urlParams}`;

  // @ts-ignore
  window.location = newLocation;
}

/**
 * @template T
 * @param {AsyncIterable<T>} iteratorObjectA
 * @param {AsyncIterable<T>} iteratorObjectB
 * @returns {AsyncIterable<[T, T]>}
 */
export async function* combineAsyncIterators(iteratorObjectA, iteratorObjectB) {
  const iteratorA = iteratorObjectA[Symbol.asyncIterator]();
  const iteratorB = iteratorObjectB[Symbol.asyncIterator]();

  while (true) {
    const promiseA = iteratorA.next();
    const promiseB = iteratorB.next();
    const nextA = await promiseA;
    const nextB = await promiseB;
    if (nextA.done !== nextB.done) {
      console.error(new Error('The iterators were not the same length'));
    }
    if (nextA.done || nextB.done) {
      return;
    }
    yield [nextA.value, nextB.value];
  }
}
