
export interface CreateElementOptions<T> {
  style: Partial<CSSStyleDeclaration>,
  parent: Element,
  children: Node | string | Array<string | Node> | [],
  href: string,
  className: string,
  title: string,
  onClick:(this: HTMLElement, event: MouseEvent) => unknown
}
