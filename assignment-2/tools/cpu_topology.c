/* Prints the logical-processor topology of this machine as JSON, using
 * GetSystemCpuSetInformation. On Intel hybrid CPUs, performance cores (P-cores)
 * report a higher EfficiencyClass than efficiency cores (E-cores). */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    ULONG length = 0;
    HANDLE process = GetCurrentProcess();
    GetSystemCpuSetInformation(NULL, 0, &length, process, 0);
    if (length == 0) {
        fprintf(stderr, "GetSystemCpuSetInformation returned no data\n");
        return 1;
    }
    unsigned char *buffer = malloc(length);
    if (buffer == NULL) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    if (!GetSystemCpuSetInformation((PSYSTEM_CPU_SET_INFORMATION)buffer, length,
                                    &length, process, 0)) {
        fprintf(stderr, "GetSystemCpuSetInformation failed: %lu\n",
                GetLastError());
        free(buffer);
        return 1;
    }

    printf("{\n  \"source\": \"GetSystemCpuSetInformation\",\n");
    printf("  \"logical_processors\": [\n");
    int first = 1;
    ULONG offset = 0;
    while (offset < length) {
        PSYSTEM_CPU_SET_INFORMATION info =
            (PSYSTEM_CPU_SET_INFORMATION)(buffer + offset);
        if (info->Type == CpuSetInformation) {
            printf("%s    {\"cpu_set_id\": %lu, \"group\": %u, "
                   "\"logical_processor_index\": %u, \"core_index\": %u, "
                   "\"last_level_cache_index\": %u, \"numa_node_index\": %u, "
                   "\"efficiency_class\": %u}",
                   first ? "" : ",\n", info->CpuSet.Id, info->CpuSet.Group,
                   info->CpuSet.LogicalProcessorIndex,
                   info->CpuSet.CoreIndex, info->CpuSet.LastLevelCacheIndex,
                   info->CpuSet.NumaNodeIndex, info->CpuSet.EfficiencyClass);
            first = 0;
        }
        offset += info->Size;
    }
    printf("\n  ]\n}\n");
    free(buffer);
    return 0;
}
