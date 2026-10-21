/*
 * Prints the CPU caches reported by Windows (GetLogicalProcessorInformationEx,
 * RelationCache) as JSON: level, type, size and the logical processors sharing it.
 *
 * Build: gcc -O2 -std=c11 -o tools/build/cache_info.exe tools/cache_info.c
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

static const char *cache_type(PROCESSOR_CACHE_TYPE type) {
    switch (type) {
    case CacheData: return "data";
    case CacheInstruction: return "instruction";
    case CacheUnified: return "unified";
    default: return "trace";
    }
}

int main(void) {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationCache, NULL, &length);
    char *buffer = malloc(length);
    if (buffer == NULL ||
        !GetLogicalProcessorInformationEx(RelationCache,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer, &length)) {
        fprintf(stderr, "GetLogicalProcessorInformationEx failed: %lu\n", GetLastError());
        return 1;
    }
    printf("[\n");
    int first = 1;
    for (DWORD offset = 0; offset < length;) {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer + offset);
        CACHE_RELATIONSHIP *cache = &info->Cache;
        printf("%s  {\"level\": %u, \"type\": \"%s\", \"size_bytes\": %lu, "
               "\"line_size\": %u, \"processor_mask\": \"0x%llx\"}",
               first ? "" : ",\n", cache->Level, cache_type(cache->Type),
               cache->CacheSize, cache->LineSize,
               (unsigned long long)cache->GroupMask.Mask);
        first = 0;
        offset += info->Size;
    }
    printf("\n]\n");
    free(buffer);
    return 0;
}
