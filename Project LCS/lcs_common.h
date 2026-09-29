#ifndef LCS_COMMON_H
#define LCS_COMMON_H

#include <stdio.h>
#include <stdlib.h>

/*
 * load_arg_or_file: benchmark-input helper.
 *
 * All four LCS variants share this parser so the correctness fixture
 * (short argv strings like "ASTRONOMY") and the benchmark fixture
 * (10,000-char randomly-generated strings) go through the same code
 * path in each program.
 *
 * Convention:
 *   - If the argument begins with '@', the rest is a filename to read.
 *   - Otherwise the argument is used verbatim as the input string.
 *
 * On file mode, trailing whitespace / newlines are stripped so file-fed
 * and argv-fed strings compare identically. Sets *is_alloc = 1 when
 * memory was allocated (caller must free); 0 when the returned pointer
 * is the original argv pointer.
 */
static inline char *load_arg_or_file(const char *arg, int *is_alloc) {
    *is_alloc = 0;
    if (arg == NULL) return NULL;
    if (arg[0] != '@') return (char *)arg;

    const char *path = arg + 1;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "load_arg_or_file: cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fprintf(stderr, "load_arg_or_file: ftell failed on %s\n", path);
        fclose(f);
        exit(1);
    }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        fprintf(stderr, "load_arg_or_file: malloc failed\n");
        exit(1);
    }
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);

    /* Strip trailing whitespace so a text file with a training newline
       yields the same string as the same content passed literally. */
    while (got > 0 && (buf[got-1] == '\n' || buf[got-1] == '\r'
                       || buf[got-1] == ' '  || buf[got-1] == '\t')) {
        got--;
    }
    buf[got] = '\0';
    *is_alloc = 1;
    return buf;
}

#endif
