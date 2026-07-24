import { api } from '@/services/api';
import { useAsync } from '@/hooks/useAsync';
import { useClassStore } from '@/stores/classStore';
import { ClassPicker } from '@/components/ClassPicker';
import { PageHeader, Spinner, ErrorState } from '@/components/ui';
import type { ClassesResponse } from '@/types';
import { useEffect } from 'react';

export default function ClassSelectorPage() {
  const setCatalog = useClassStore((s) => s.setCatalog);
  const loaded = useClassStore((s) => s.loaded);
  const { data, error, loading, reload } = useAsync<ClassesResponse>((s) => api.classes(s), []);

  useEffect(() => {
    if (data) setCatalog(data.classes, data.groups);
  }, [data, setCatalog]);

  return (
    <div>
      <PageHeader
        title="Class Selector"
        subtitle="Choose which of the 80 COCO classes the detector reports. Your selection is saved locally and applied to the Live feed as allowed_class_ids."
      />
      {loading && !loaded && <Spinner label="Loading class catalog…" />}
      {error && !loaded && <ErrorState message={error} onRetry={reload} />}
      {loaded && (
        <div className="card card-pad">
          <ClassPicker />
        </div>
      )}
    </div>
  );
}
