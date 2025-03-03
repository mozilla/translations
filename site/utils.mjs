/**
 * @param {any} any
 * @returns {any}
 */
export function asAny(any) {
  return any;
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
 * @param {string} key
 * @param {any} value
 */
export function exposeAsGlobal(key, value) {
  console.log(key, value);
  asAny(window)[key] = value;
}
