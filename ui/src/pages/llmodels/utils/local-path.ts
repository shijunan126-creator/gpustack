import { DirectoryEntry } from '../config/types';

/**
 * What to list for what has been typed: the directory the text sits in, and
 * the part of the next segment already written, which narrows the list.
 * Text that is not an absolute path lists the root.
 */
export const listTarget = (value: string): { dir: string; prefix: string } => {
  const text = (value || '').trim();
  if (!text.startsWith('/')) {
    return { dir: '/', prefix: text };
  }
  const cut = text.lastIndexOf('/');
  return { dir: text.slice(0, cut + 1) || '/', prefix: text.slice(cut + 1) };
};

export interface PathOption {
  value: string;
  name: string;
  isModel: boolean;
  /** A model is a place to stop; any other directory is one to go into. */
  isFinal: boolean;
}

/**
 * The entries that continue the typed text. Picking a directory that is not a
 * model adds a trailing slash, so the next list opens inside it.
 */
export const toPathOptions = (
  entries: DirectoryEntry[],
  prefix: string
): PathOption[] => {
  const lowered = prefix.toLowerCase();
  return entries
    .filter((entry) => entry.name.toLowerCase().startsWith(lowered))
    .map((entry) => {
      const isFinal = entry.is_model || !entry.is_dir;
      return {
        value: isFinal ? entry.path : `${entry.path}/`,
        name: entry.is_dir ? `${entry.name}/` : entry.name,
        isModel: entry.is_model,
        isFinal
      };
    });
};
