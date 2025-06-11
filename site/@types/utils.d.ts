/**
 * The interface used by the createElements utility.
 */
export interface CreateElementOptions {
  style: Partial<CSSStyleDeclaration>,
  parent: Element,
  children: Node | string | Array<string | Node> | [],
  href: string,
  className: string,
  title: string,
  onClick:(this: HTMLElement, event: MouseEvent) => unknown
}
