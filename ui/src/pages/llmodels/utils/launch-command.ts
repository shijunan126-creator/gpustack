/**
 * Reads an engine launch script — `export` lines plus a `vllm serve ...` or
 * `python -m sglang.launch_server ...` command, as engine docs and recipes
 * publish them — into the two things a deployment form holds: backend
 * parameters and environment variables.
 *
 * Nothing is executed or expanded. A `$VAR` stays text and is reported, since
 * its value lives in the author's shell and not in the script.
 */

export interface DroppedFlag {
  flag: string;
  value?: string;
  /** `managed`: GPUStack sets it per instance. `model`: the model source does. */
  reason: 'managed' | 'model';
}

export interface LaunchCommandResult {
  env: Record<string, string>;
  /** Entries in the shape the parameter list holds: `--key=value`, `--flag`. */
  parameters: string[];
  /** The model the command names, if any. Informational; never applied. */
  model?: string;
  dropped: DroppedFlag[];
  /** Names of the flags and variables whose value reads a shell variable. */
  variables: string[];
  /** Set when nothing usable was found. */
  error?: 'empty' | 'no_command' | 'unterminated_quote';
}

// GPUStack passes these itself and, because it only fills in the ones that are
// missing, a pasted value would win and break routing to the instance.
const MANAGED_FLAGS = new Set(['--host', '--port', '--served-model-name']);
const MODEL_FLAGS = new Set(['--model', '--model-path']);

interface Word {
  text: string;
  /** True when a `$` reached the text unquoted or inside double quotes. */
  expands: boolean;
}

type Statement = Word[];

const isNumber = (token: string): boolean =>
  token.trim() !== '' && !Number.isNaN(Number(token));

const looksLikeFlag = (token: string): boolean =>
  token.startsWith('-') && token.length > 1 && !isNumber(token);

/**
 * Splits shell text into statements of words. Understands quotes, backslash
 * escapes and continuations, `#` comments, and `;`, `&&`, `||` and newlines as
 * separators. Returns null for an unterminated quote.
 */
export const tokenize = (input: string): Statement[] | null => {
  const text = input.replace(/\r\n?/g, '\n');
  const statements: Statement[] = [];
  let words: Statement = [];
  let current = '';
  let expands = false;
  let inWord = false;

  const endWord = () => {
    if (inWord) {
      words.push({ text: current, expands });
    }
    current = '';
    expands = false;
    inWord = false;
  };
  const endStatement = () => {
    endWord();
    if (words.length) {
      statements.push(words);
    }
    words = [];
  };

  let i = 0;
  while (i < text.length) {
    const c = text[i];
    if (c === '\\') {
      if (text[i + 1] === '\n') {
        i += 2;
        continue;
      }
      if (i + 1 < text.length) {
        current += text[i + 1];
        inWord = true;
        i += 2;
        continue;
      }
      i += 1;
      continue;
    }
    if (c === "'") {
      const end = text.indexOf("'", i + 1);
      if (end < 0) {
        return null;
      }
      current += text.slice(i + 1, end);
      inWord = true;
      i = end + 1;
      continue;
    }
    if (c === '"') {
      inWord = true;
      i += 1;
      let closed = false;
      while (i < text.length) {
        const d = text[i];
        if (d === '"') {
          closed = true;
          i += 1;
          break;
        }
        if (d === '\\' && '"\\$`\n'.includes(text[i + 1] ?? '')) {
          if (text[i + 1] !== '\n') {
            current += text[i + 1];
          }
          i += 2;
          continue;
        }
        if (d === '$') {
          expands = true;
        }
        current += d;
        i += 1;
      }
      if (!closed) {
        return null;
      }
      continue;
    }
    if (c === '#' && !inWord) {
      while (i < text.length && text[i] !== '\n') {
        i += 1;
      }
      continue;
    }
    if (c === '\n' || c === ';') {
      endStatement();
      i += 1;
      continue;
    }
    if ((c === '&' || c === '|') && text[i + 1] === c) {
      endStatement();
      i += 2;
      continue;
    }
    if (c === ' ' || c === '\t') {
      endWord();
      i += 1;
      continue;
    }
    if (c === '$') {
      expands = true;
    }
    current += c;
    inWord = true;
    i += 1;
  }
  endStatement();
  return statements;
};

const ASSIGNMENT = /^([A-Za-z_][A-Za-z0-9_]*)=([\s\S]*)$/;

// Words that may sit in front of the launcher without changing what it runs.
const WRAPPERS = new Set(['nohup', 'exec', 'setsid', 'time']);

const baseName = (path: string): string => path.split('/').pop() || path;

/**
 * The words after the launcher (`vllm serve`, `python -m vllm.entrypoints...`,
 * `sglang serve`, `python -m sglang.launch_server`), or null when the
 * statement launches something else.
 */
const engineArguments = (words: string[]): string[] | null => {
  let start = 0;
  while (start < words.length && WRAPPERS.has(words[start])) {
    start += 1;
  }
  const [program, second, third] = words.slice(start);
  if (!program) {
    return null;
  }
  const name = baseName(program);
  if ((name === 'vllm' || name === 'sglang') && second === 'serve') {
    return words.slice(start + 2);
  }
  if (/^python[\d.]*$/.test(name) && second === '-m' && third) {
    if (/^(vllm|sglang)(\.|$)/.test(third)) {
      const rest = words.slice(start + 3);
      return rest[0] === 'serve' ? rest.slice(1) : rest;
    }
  }
  return null;
};

// `{ "a": 1 }` spread over several lines is one value; keep it compact when it
// is valid JSON and leave anything else as written.
const compactJson = (value: string): string => {
  const trimmed = value.trim();
  if (!/^[{[]/.test(trimmed)) {
    return value;
  }
  try {
    return JSON.stringify(JSON.parse(trimmed));
  } catch {
    return trimmed;
  }
};

interface FlagGroup {
  flag: string;
  values: Word[];
  inline: boolean;
}

const groupFlags = (args: Word[]): FlagGroup[] => {
  const groups: FlagGroup[] = [];
  args.forEach((arg) => {
    if (looksLikeFlag(arg.text)) {
      const eq = arg.text.indexOf('=');
      if (eq > 0) {
        groups.push({
          flag: arg.text.slice(0, eq),
          values: [{ text: arg.text.slice(eq + 1), expands: arg.expands }],
          inline: true
        });
      } else {
        groups.push({ flag: arg.text, values: [], inline: false });
      }
    } else if (groups.length && !groups[groups.length - 1].inline) {
      groups[groups.length - 1].values.push(arg);
    }
  });
  return groups;
};

/** One flag and its values, as the parameter list stores them. */
const toEntries = ({ flag, values, inline }: FlagGroup): string[] => {
  if (!values.length) {
    return [flag];
  }
  if (values.length === 1 || inline) {
    return [`${flag}=${compactJson(values[0].text)}`];
  }
  // A flag that takes several values keeps them as separate tokens, so none is
  // read as a flag of its own.
  return [flag, ...values.map((v) => compactJson(v.text))];
};

export const parseLaunchCommand = (input: string): LaunchCommandResult => {
  const result: LaunchCommandResult = {
    env: {},
    parameters: [],
    dropped: [],
    variables: []
  };
  if (!input.trim()) {
    return { ...result, error: 'empty' };
  }
  const statements = tokenize(input);
  if (!statements) {
    return { ...result, error: 'unterminated_quote' };
  }

  let launched = false;
  statements.forEach((statement) => {
    let index = 0;
    if (statement[0]?.text === 'export' || statement[0]?.text === 'declare') {
      // `export A=1 B="2"`; a name with no value only marks it exported.
      const assignments = statement
        .slice(1)
        .filter((word) => word.text !== '-x');
      assignments.forEach((word) => {
        const match = ASSIGNMENT.exec(word.text);
        if (match) {
          result.env[match[1]] = match[2];
          if (word.expands) {
            result.variables.push(match[1]);
          }
        }
      });
      return;
    }
    // `NAME=value engine ...` scopes the variables to the command.
    const scoped: Record<string, string> = {};
    const scopedVariables: string[] = [];
    while (index < statement.length && ASSIGNMENT.test(statement[index].text)) {
      const match = ASSIGNMENT.exec(statement[index].text)!;
      scoped[match[1]] = match[2];
      if (statement[index].expands) {
        scopedVariables.push(match[1]);
      }
      index += 1;
    }
    const rest = statement.slice(index);
    if (!rest.length) {
      Object.assign(result.env, scoped);
      result.variables.push(...scopedVariables);
      return;
    }
    const args = engineArguments(rest.map((word) => word.text));
    if (!args || launched) {
      return;
    }
    launched = true;
    Object.assign(result.env, scoped);
    result.variables.push(...scopedVariables);

    const words = rest.slice(rest.length - args.length);
    let flagsFrom = 0;
    if (words.length && !looksLikeFlag(words[0].text)) {
      result.model = words[0].text;
      flagsFrom = 1;
    }
    groupFlags(words.slice(flagsFrom)).forEach((group) => {
      const value = group.values.map((v) => v.text).join(' ') || undefined;
      if (MANAGED_FLAGS.has(group.flag)) {
        result.dropped.push({ flag: group.flag, value, reason: 'managed' });
        return;
      }
      if (MODEL_FLAGS.has(group.flag)) {
        result.model = value;
        result.dropped.push({ flag: group.flag, value, reason: 'model' });
        return;
      }
      if (group.values.some((v) => v.expands)) {
        result.variables.push(group.flag);
      }
      result.parameters.push(...toEntries(group));
    });
  });

  if (!launched) {
    return { ...result, error: 'no_command' };
  }
  return result;
};

/** The flag an existing parameter entry sets, if it starts with one. */
const entryFlag = (entry: string): string | null => {
  const first = entry.trim().split(/[\s=]/, 1)[0];
  return looksLikeFlag(first) ? first : null;
};

/**
 * Folds a parsed command into the form's current values. A flag that is
 * already set is replaced by the pasted one rather than repeated; variables of
 * the same name take the pasted value; everything else stays as it was.
 */
export const mergeLaunchCommand = (
  current: { parameters?: string[] | null; env?: Record<string, any> | null },
  parsed: Pick<LaunchCommandResult, 'parameters' | 'env'>
): { parameters: string[]; env: Record<string, any> } => {
  const pasted = new Set(
    parsed.parameters
      .map((entry) => entryFlag(entry))
      .filter((flag): flag is string => !!flag)
  );
  const kept = (current.parameters || []).filter((entry) => {
    const flag = entryFlag(entry);
    return !flag || !pasted.has(flag);
  });
  return {
    parameters: [...kept, ...parsed.parameters],
    env: { ...(current.env || {}), ...parsed.env }
  };
};
