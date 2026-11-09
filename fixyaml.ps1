$p = 'gpustack/assets/profiles_config/profiles_config.yaml'
$c = Get-Content $p -Raw
$c = $c.Replace("    dataset_name: Random`n    dataset_input_tokens: 1024 # vLLM Random default", "    dataset_name: GSM8K`n    dataset_input_tokens: 1024 # ignored by GSM8K: its prompts are the dataset's own")
$c = $c.Replace("    dataset_name: Random`n    dataset_input_tokens: 128", "    dataset_name: GSM8K`n    dataset_input_tokens: 128 # ignored by GSM8K: its prompts are the dataset's own")
Set-Content -Path $p -Value $c -NoNewline