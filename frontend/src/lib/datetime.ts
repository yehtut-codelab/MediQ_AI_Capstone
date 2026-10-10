// Helpers for preloaded <input type="datetime-local"> values. All values are
// browser-local wall time in the "YYYY-MM-DDTHH:mm" form the input expects.

const pad = (n: number) => String(n).padStart(2, "0");

function toLocalInput(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** The current date and time. */
export function nowLocal(): string {
  return toLocalInput(new Date());
}

/** Today's date at the given "HH:mm" time of day. */
export function todayAt(hhmm: string): string {
  const [h, mi] = hhmm.split(":").map(Number);
  const now = new Date();
  return toLocalInput(new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, mi));
}

/** The next occurrence of the given time of day that is not in the past:
 *  today if that time is still ahead, otherwise tomorrow. `iso` is any
 *  "YYYY-MM-DDTHH:mm…" timestamp; only its time of day is used. */
export function upcomingAt(iso: string): string {
  const [h, mi] = iso.slice(11, 16).split(":").map(Number);
  const now = new Date();
  const t = new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, mi);
  if (t < now) t.setDate(t.getDate() + 1);
  return toLocalInput(t);
}
