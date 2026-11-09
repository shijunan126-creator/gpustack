$p = "ui/src/pages/benchmark/forms/random-settings.tsx"
$c = [IO.File]::ReadAllText($p)
$old = "dataset_input_tokens:`r`n                  picked?.format === 'random'`r`n                    ? (form.getFieldValue('dataset_input_tokens') ?? 1024)`r`n                    : null,"
$new = "dataset_input_tokens:`r`n                  picked?.format === 'sharegpt'`r`n                    ? null`r`n                    : (form.getFieldValue('dataset_input_tokens') ?? 1024),"
if ($c.Contains($old)) { $c = $c.Replace($old, $new); [IO.File]::WriteAllText($p, $c); "updated (crlf)" } else { "pattern not found" }