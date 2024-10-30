#pragma once

#define MSG(fmt, ...) fprintf(stderr, "%s(%d) %s(): " fmt "\n", __FILE__, __LINE__, __func__, ##__VA_ARGS__)

