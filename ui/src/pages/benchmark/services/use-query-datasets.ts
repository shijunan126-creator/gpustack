import { useState } from 'react';
import { queryBenchmarkDatasets } from '../apis';
import { BenchmarkDatasetItem, BenchmarkDatasetOption } from '../config/types';

// The dataset list the create form selects from. Fetched rather than imported:
// an uploaded corpus is a row, and the form must be able to pick one that did
// not exist when the UI was built.
export default function useQueryDatasets() {
  const [datasets, setDatasets] = useState<BenchmarkDatasetItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchDatasets = async (): Promise<BenchmarkDatasetItem[]> => {
    setLoading(true);
    try {
      const res = await queryBenchmarkDatasets();
      const items = res?.items ?? [];
      setDatasets(items);
      return items;
    } catch {
      return [];
    } finally {
      setLoading(false);
    }
  };

  const options: BenchmarkDatasetOption[] = datasets.map((item) => ({
    value: item.name,
    label: item.name,
    format: item.format,
    has_answers: item.has_answers,
    sample_count: item.sample_count ?? null,
    description: item.description ?? ''
  }));

  return { datasets, options, loading, fetchDatasets };
}
