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

/**
 * Gets an element and throws if it doesn't exists. The className provided to specialize
 * the type.
 *
 * @template {typeof HTMLElement} T
 *
 * @param {string} id
 * @param {{ new (): InstanceType<T>; prototype: InstanceType<T>; }} [className]
 * @returns {InstanceType<T>}
 */
export function getElement(
  id,
  // @ts-expect-error - This is a bit tricky, but the `type HTMLElement` is a little bit
  // different than the `var HTMLElement`. The `type` is an interface, while the `var`
  // contains the `new` and `prototype` properties. The `var` can be passed around by
  // value. Perhaps there is a correct way to type this, but it's not worth figuring out.
  className = HTMLElement
) {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error("Could not find element by id: " + id);
  }
  if (!(element instanceof className)) {
    throw new Error(
      `Selected element #${id} was not an instance of ${className.name}`
    );
  }
  return element;
}

