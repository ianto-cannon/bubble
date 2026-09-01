#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <dirent.h>
#include <float.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define PI 3.14159265358979323846
#define MAX_ITER 1000000
#define MAX_BINS 1000000   /* large enough for expected bin indices */

/* ------------------------------------------------------------------------- */
/* Data structures                                                           */
/* ------------------------------------------------------------------------- */

/* Structure to hold all data that was previously printed for a bin */
typedef struct {
    int valid;               /* 1 if this bin has been seen */
    double r;
    double minus_z;
    double psi;
    double dPsi;
    double capLen;
    double RadTop;
    double Volume;
    double area;
    double centroid;
    double maxRad;
} BinData;

/* Global arrays for radius and angle bins, different selection criteria */
BinData radBinsVol[MAX_BINS];
BinData angBinsVol[MAX_BINS];
BinData angBinsMaxRad[MAX_BINS];
BinData radBinsMaxAng[MAX_BINS];

/* Highest bin indices ever used, for final file writing */
int maxRadBinVolUsed = -1;
int maxAngBinVolUsed = -1;
int maxAngBinRadUsed = -1;
int maxRadBinAngUsed = -1;

/* ------------------------------------------------------------------------- */
/* Forward declarations                                                      */
/* ------------------------------------------------------------------------- */
void write_bin_file(const char *filename, BinData *bins, int max_bin_used);
int AdamsBashforthProfile(double capLen, double RadTop,
                          double contactAng, const char *fname,
                          double angleSave, double radSave,
                          double angleSaveLarge, double radSaveLarge,
                          double *outVolume, double *out_r, double *out_z,
                          double *out_centroid, double *out_psi,
                          int *out_i);

/* ------------------------------------------------------------------------- */
/* Grid-based nearest neighbour routines                                     */
/* ------------------------------------------------------------------------- */

typedef struct {
    double dist;
    int idx;
} Neighbor;

typedef struct {
    const double *x, *y;    // point coordinates (not owned)
    double min_x, min_y, max_x, max_y;
    double cell_size;
    int nx, ny;
    int *head;              // head of linked list for each cell (size nx*ny)
    int *next;              // next pointer for each point (size N)
} Grid;

static void build_grid(Grid *g, const double *x, const double *y, int N) {
    double minx = x[0], maxx = x[0], miny = y[0], maxy = y[0];
    for (int i = 1; i < N; ++i) {
        if (x[i] < minx) minx = x[i];
        if (x[i] > maxx) maxx = x[i];
        if (y[i] < miny) miny = y[i];
        if (y[i] > maxy) maxy = y[i];
    }
    if (maxx == minx) maxx = minx + 1e-12;
    if (maxy == miny) maxy = miny + 1e-12;

    double area = (maxx - minx) * (maxy - miny);
    double cell = sqrt(area / N);
    if (cell <= 0.0) cell = 1e-6;
    cell *= 1.5;

    g->min_x = minx;
    g->min_y = miny;
    g->max_x = maxx;
    g->max_y = maxy;
    g->cell_size = cell;
    g->nx = (int)ceil((maxx - minx) / cell);
    g->ny = (int)ceil((maxy - miny) / cell);
    if (g->nx < 1) g->nx = 1;
    if (g->ny < 1) g->ny = 1;

    int total_cells = g->nx * g->ny;
    g->head = calloc(total_cells, sizeof(int));
    g->next = malloc(N * sizeof(int));
    if (!g->head || !g->next) {
        fprintf(stderr, "Memory allocation failed in build_grid\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < total_cells; ++i) g->head[i] = -1;

    for (int i = 0; i < N; ++i) {
        int cx = (int)floor((x[i] - minx) / cell);
        int cy = (int)floor((y[i] - miny) / cell);
        if (cx < 0) cx = 0;
        if (cx >= g->nx) cx = g->nx - 1;
        if (cy < 0) cy = 0;
        if (cy >= g->ny) cy = g->ny - 1;
        int cell_idx = cy * g->nx + cx;
        g->next[i] = g->head[cell_idx];
        g->head[cell_idx] = i;
    }

    g->x = x;
    g->y = y;
}

static void free_grid(Grid *g) {
    free(g->head);
    free(g->next);
}

static double ring_min_dist_sq(const Grid *g, double qx, double qy, int r) {
    double min_d2 = HUGE_VAL;
    int qcx = (int)floor((qx - g->min_x) / g->cell_size);
    int qcy = (int)floor((qy - g->min_y) / g->cell_size);

    for (int dx = -r; dx <= r; ++dx) {
        for (int dy = -r; dy <= r; ++dy) {
            if (abs(dx) != r && abs(dy) != r) continue;
            int cx = qcx + dx;
            int cy = qcy + dy;
            if (cx < 0 || cx >= g->nx || cy < 0 || cy >= g->ny) continue;

            double cell_xmin = g->min_x + cx * g->cell_size;
            double cell_xmax = cell_xmin + g->cell_size;
            double cell_ymin = g->min_y + cy * g->cell_size;
            double cell_ymax = cell_ymin + g->cell_size;

            double cx_clamped = qx < cell_xmin ? cell_xmin : (qx > cell_xmax ? cell_xmax : qx);
            double cy_clamped = qy < cell_ymin ? cell_ymin : (qy > cell_ymax ? cell_ymax : qy);

            double d2 = (cx_clamped - qx) * (cx_clamped - qx) +
                        (cy_clamped - qy) * (cy_clamped - qy);
            if (d2 < min_d2) min_d2 = d2;
        }
    }
    return min_d2;
}

static int knn_grid(const Grid *g, const unsigned char *used,
                    double qx, double qy, int k, int *out_idx) {
    if (k <= 0 || !g->head) return 0;

    int qcx = (int)floor((qx - g->min_x) / g->cell_size);
    int qcy = (int)floor((qy - g->min_y) / g->cell_size);
    if (qcx < 0) qcx = 0;
    if (qcx >= g->nx) qcx = g->nx - 1;
    if (qcy < 0) qcy = 0;
    if (qcy >= g->ny) qcy = g->ny - 1;

    Neighbor *heap = malloc(k * sizeof(Neighbor));
    if (!heap) return 0;
    int heap_size = 0;
    int r = 0;

    while (1) {
        int any_cell_processed = 0;

        for (int dx = -r; dx <= r; ++dx) {
            for (int dy = -r; dy <= r; ++dy) {
                if (abs(dx) != r && abs(dy) != r) continue;
                int cx = qcx + dx;
                int cy = qcy + dy;
                if (cx < 0 || cx >= g->nx || cy < 0 || cy >= g->ny) continue;
                any_cell_processed = 1;

                int cell_idx = cy * g->nx + cx;
                int idx = g->head[cell_idx];
                while (idx != -1) {
                    if (!used[idx]) {
                        double dxp = g->x[idx] - qx;
                        double dyp = g->y[idx] - qy;
                        double dist = dxp * dxp + dyp * dyp;

                        if (heap_size < k) {
                            heap[heap_size].dist = dist;
                            heap[heap_size].idx = idx;
                            heap_size++;
                            int i = heap_size - 1;
                            while (i > 0) {
                                int parent = (i - 1) / 2;
                                if (heap[parent].dist < heap[i].dist) {
                                    Neighbor tmp = heap[parent];
                                    heap[parent] = heap[i];
                                    heap[i] = tmp;
                                    i = parent;
                                } else break;
                            }
                        } else if (dist < heap[0].dist) {
                            heap[0].dist = dist;
                            heap[0].idx = idx;
                            int i = 0;
                            while (1) {
                                int left = 2 * i + 1, right = 2 * i + 2, largest = i;
                                if (left < heap_size && heap[left].dist > heap[largest].dist) largest = left;
                                if (right < heap_size && heap[right].dist > heap[largest].dist) largest = right;
                                if (largest == i) break;
                                Neighbor tmp = heap[i];
                                heap[i] = heap[largest];
                                heap[largest] = tmp;
                                i = largest;
                            }
                        }
                    }
                    idx = g->next[idx];
                }
            }
        }

        if (heap_size >= k) {
            double next_min_dist_sq = ring_min_dist_sq(g, qx, qy, r + 1);
            if (next_min_dist_sq > heap[0].dist) break;
        }

        r++;
        if (!any_cell_processed && heap_size == 0) break;
        if (r > g->nx + g->ny) break;
    }

    int cnt = heap_size;
    for (int i = 0; i < cnt; ++i) {
        out_idx[i] = heap[i].idx;
    }
    free(heap);
    return cnt;
}

/* ------------------------------------------------------------------------- */
/* File utilities                                                            */
/* ------------------------------------------------------------------------- */

static int parse_line_doubles(const char *line, double *out, int max_cols) {
    int n = 0;
    const char *p = line;
    while (n < max_cols && *p) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;
        char *end;
        double v = strtod(p, &end);
        if (end == p) break;
        out[n++] = v;
        p = end;
    }
    return n;
}

static int load_txt(const char *path, double **out_data, int *out_rows, int *out_cols) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    int rows = 0, cols = 0;
    char *line = NULL; size_t cap = 0; ssize_t len;
    double *buf = NULL; int buf_cap = 0;

    while ((len = getline(&line, &cap, f)) != -1) {
        int c = 0;
        const char *p = line;
        while (*p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            char *end; strtod(p, &end);
            if (end == p) break;
            c++;
            p = end;
        }
        if (c == 0) continue;
        if (c > cols) cols = c;
        if (rows + 1 > buf_cap) {
            buf_cap = buf_cap ? buf_cap * 2 : 1024;
            buf = realloc(buf, (size_t)buf_cap * cols * sizeof(double));
            if (!buf) { fclose(f); free(line); return -1; }
        }
        double tmp[64];
        int n = parse_line_doubles(line, tmp, (c < 64 ? c : 64));
        for (int k = 0; k < n; ++k) buf[(size_t)rows * cols + k] = tmp[k];
        for (int k = n; k < cols; ++k) buf[(size_t)rows * cols + k] = 0.0;
        rows++;
    }
    fclose(f);
    free(line);

    if (rows == 0 || cols == 0) { free(buf); return -1; }
    *out_data = buf; *out_rows = rows; *out_cols = cols;
    return 0;
}

static int save_txt(const char *path, const double *data, int rows, int cols) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(f, "%.10e%s", data[(size_t)i * cols + j],
                    (j == cols - 1) ? "\n" : " ");
        }
    }
    fclose(f);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Adams-Bashforth profile integration                                       */
/* ------------------------------------------------------------------------- */
int AdamsBashforthProfile(double capLen, double RadTop,
                          double contactAng, const char *fname,
                          double angleSave, double radSave,
                          double angleSaveLarge, double radSaveLarge,
                          double *outVolume, double *out_r, double *out_z,
                          double *out_centroid, double *out_psi,
                          int *out_i)
{
    double ds = fmin(fmin(1e-5 * fabs(RadTop), 1e-5 * fabs(capLen)), 1e-5);
    double psi = 0.0, r = 0.0, z = 0.0;
    double Volume = 0.0, centroid = 0.0, area = 0.0;
    double dPsiPrev = 0.0, maxRad = 0.0;
    double drPrev = 0.0, dzPrev = 0.0;
    int i;

    FILE *adams_txt = NULL;
    if (fname) adams_txt = fopen(fname, "w");

    for (i = 0; i < MAX_ITER; ++i) {
        double dr = ds * cos(psi);
        double dz = ds * sin(psi);
        if (i == 0) { drPrev = dr; dzPrev = dz; }

        r += 1.5 * dr - 0.5 * drPrev;
        z += 1.5 * dz - 0.5 * dzPrev;
        drPrev = dr;
        dzPrev = dz;

        double sinpsi_over_r = (r > 0.0) ? (sin(psi) / r) : 0.0;
        double dPsi = ds * (2.0 / RadTop - z / (capLen * capLen) - sinpsi_over_r);
        psi += 1.5 * dPsi - 0.5 * dPsiPrev;

        Volume  += PI * r * r * dz;
        centroid+= z * PI * r * r * dz;
        area    += 2.0 * PI * r * ds;
        if (r > maxRad) maxRad = r;

        if (adams_txt) {
            if ((i % 100 == 0) || (dPsi > 1e-2)) {
                fprintf(adams_txt,
                        "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
                        r, -z, psi, dPsi, capLen, RadTop, Volume, area,
                        (Volume != 0.0 ? centroid / Volume : 0.0), maxRad);
            }
        }

        /* Fine angle bin handling (for consolidated max) */
        if (angleSave > 0.0 && i > 0) {
            int b = (int)floor(psi / PI / angleSave);
            if (b >= 0 && b < MAX_BINS) {
                /* 1. Max volume for angle bins */
                if (!angBinsVol[b].valid || Volume > angBinsVol[b].Volume) {
                    angBinsVol[b].valid = 1;
                    angBinsVol[b].r = r;
                    angBinsVol[b].minus_z = -z;
                    angBinsVol[b].psi = psi;
                    angBinsVol[b].dPsi = dPsi;
                    angBinsVol[b].capLen = capLen;
                    angBinsVol[b].RadTop = RadTop;
                    angBinsVol[b].Volume = Volume;
                    angBinsVol[b].area = area;
                    angBinsVol[b].centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
                    angBinsVol[b].maxRad = maxRad;
                    if (b > maxAngBinVolUsed) maxAngBinVolUsed = b;
                }
                /* 2. Max radius for angle bins */
                if (!angBinsMaxRad[b].valid || r > angBinsMaxRad[b].r) {
                    angBinsMaxRad[b].valid = 1;
                    angBinsMaxRad[b].r = r;
                    angBinsMaxRad[b].minus_z = -z;
                    angBinsMaxRad[b].psi = psi;
                    angBinsMaxRad[b].dPsi = dPsi;
                    angBinsMaxRad[b].capLen = capLen;
                    angBinsMaxRad[b].RadTop = RadTop;
                    angBinsMaxRad[b].Volume = Volume;
                    angBinsMaxRad[b].area = area;
                    angBinsMaxRad[b].centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
                    angBinsMaxRad[b].maxRad = maxRad;
                    if (b > maxAngBinRadUsed) maxAngBinRadUsed = b;
                }
            }
        }

        /* Fine radius bin handling (for consolidated max) */
        if (radSave > 0.0 && i > 0) {
            int b = (int)floor(r / radSave);
            if (b >= 0 && b < MAX_BINS) {
                /* 1. Max volume for radius bins */
                if (!radBinsVol[b].valid || Volume > radBinsVol[b].Volume) {
                    radBinsVol[b].valid = 1;
                    radBinsVol[b].r = r;
                    radBinsVol[b].minus_z = -z;
                    radBinsVol[b].psi = psi;
                    radBinsVol[b].dPsi = dPsi;
                    radBinsVol[b].capLen = capLen;
                    radBinsVol[b].RadTop = RadTop;
                    radBinsVol[b].Volume = Volume;
                    radBinsVol[b].area = area;
                    radBinsVol[b].centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
                    radBinsVol[b].maxRad = maxRad;
                    if (b > maxRadBinVolUsed) maxRadBinVolUsed = b;
                }
                /* 2. Max angle for radius bins */
                if (!radBinsMaxAng[b].valid || psi > radBinsMaxAng[b].psi) {
                    radBinsMaxAng[b].valid = 1;
                    radBinsMaxAng[b].r = r;
                    radBinsMaxAng[b].minus_z = -z;
                    radBinsMaxAng[b].psi = psi;
                    radBinsMaxAng[b].dPsi = dPsi;
                    radBinsMaxAng[b].capLen = capLen;
                    radBinsMaxAng[b].RadTop = RadTop;
                    radBinsMaxAng[b].Volume = Volume;
                    radBinsMaxAng[b].area = area;
                    radBinsMaxAng[b].centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
                    radBinsMaxAng[b].maxRad = maxRad;
                    if (b > maxRadBinAngUsed) maxRadBinAngUsed = b;
                }
            }
        }

        /* Large angle bin handling (per‑bin files) */
        if (angleSaveLarge > 0.0 && i > 0) {
            int angBinLarge     = (int)floor((psi      / PI) / angleSaveLarge);
            int angBinLargePrev = (int)floor(((psi-dPsi)/ PI) / angleSaveLarge);
            if (angBinLarge != angBinLargePrev) {
                int b = (angBinLarge > angBinLargePrev) ? angBinLarge : angBinLargePrev;
                char angFname[256];
                snprintf(angFname, sizeof(angFname), "simData/ang%05d.txt", b);
                FILE *ang_txt = fopen(angFname, "a");
                if (ang_txt) {
                    fprintf(ang_txt,
                            "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
                            r, -z, psi, dPsi, capLen, RadTop, Volume, area,
                            (Volume != 0.0 ? centroid / Volume : 0.0), maxRad);
                    fclose(ang_txt);
                }
            }
        }

        /* Large radius bin handling (per‑bin files) */
        if (radSaveLarge > 0.0 && i > 0) {
            int radBinLarge     = (int)floor(r       / radSaveLarge);
            int radBinLargePrev = (int)floor((r - dr) / radSaveLarge);
            if (radBinLarge != radBinLargePrev) {
                int b = (radBinLarge > radBinLargePrev) ? radBinLarge : radBinLargePrev;
                char radFname[256];
                snprintf(radFname, sizeof(radFname), "simData/rad%05d.txt", b);
                FILE *rad_txt = fopen(radFname, "a");
                if (rad_txt) {
                    fprintf(rad_txt,
                            "%.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
                            r, -z, psi, dPsi, capLen, RadTop, Volume, area,
                            (Volume != 0.0 ? centroid / Volume : 0.0), maxRad);
                    fclose(rad_txt);
                }
            }
        }

        if (psi < 0.0) break;
        if (psi > PI)  break;
        if (dPsi > 0.0 && dPsiPrev < 0.0) break;
        dPsiPrev = dPsi;
    }

    if (adams_txt) fclose(adams_txt);

    if (outVolume)  *outVolume  = Volume;
    if (out_r)      *out_r      = r;
    if (out_z)      *out_z      = z;
    if (out_centroid)*out_centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
    if (out_psi)    *out_psi    = psi;
    if (out_i)      *out_i      = i;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* write_bin_file                                                            */
/* ------------------------------------------------------------------------- */
void write_bin_file(const char *filename, BinData *bins, int max_bin_used)
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error opening %s for writing\n", filename);
        return;
    }
    fprintf(fp, "# bin r -z psi dPsi capLen RadTop Volume area centroid maxRad\n");
    for (int b = 0; b <= max_bin_used; b++) {
        if (bins[b].valid) {
            fprintf(fp,
                    "%d %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g %.10g\n",
                    b, bins[b].r, bins[b].minus_z, bins[b].psi,
                    bins[b].dPsi, bins[b].capLen, bins[b].RadTop,
                    bins[b].Volume, bins[b].area, bins[b].centroid,
                    bins[b].maxRad);
        }
    }
    fclose(fp);
}

/* ------------------------------------------------------------------------- */
/* reorder_drop_height_vs_vol                                                */
/* ------------------------------------------------------------------------- */
int reorder_drop_height_vs_vol(const char *nam)
{
    const char *folName = "simData/";
    DIR *dir = opendir(folName);
    if (!dir) { perror("opendir"); return -1; }

    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        const char *fname = de->d_name;

        if (strstr(fname, "loop")) continue;
        if (!strstr(fname, "txt")) continue;
        if (nam && *nam && !strstr(fname, nam)) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s%s", folName, fname);

        double *df = NULL; int N = 0, ncols = 0;
        if (load_txt(path, &df, &N, &ncols) != 0) continue;
        if (N < 2) { free(df); continue; }

        double *dfLoop = calloc((size_t)N * ncols, sizeof(double));
        for (int i = 0; i < N; ++i) dfLoop[(size_t)i * ncols + 4] = 1.0;

        double *x   = malloc(sizeof(double) * N);
        double *y   = malloc(sizeof(double) * N);
        double *vol = malloc(sizeof(double) * N);
        for (int i = 0; i < N; ++i) {
            double row1 = df[(size_t)i * ncols + 1];
            double row4 = df[(size_t)i * ncols + 4];
            double row6 = df[(size_t)i * ncols + 6];
            x[i]   = row1 / row4;
            y[i]   = cbrt(row6) / row4;
            vol[i] = row6 / (row4 * row4 * row4);
        }

        unsigned char *used = calloc(N, sizeof(unsigned char));

        // Build spatial grid for fast neighbour search
        Grid grid;
        build_grid(&grid, x, y, N);

        for (int j = 1; j < N; ++j) {
            double row1 = dfLoop[(size_t)(j-1) * ncols + 1];
            double row4 = dfLoop[(size_t)(j-1) * ncols + 4];
            double row6 = dfLoop[(size_t)(j-1) * ncols + 6];
            double x_prev   = row1 / row4;
            double y_prev   = cbrt(row6) / row4;
            double vol_prev = row6 / (row4 * row4 * row4);

            int k = (20 < N) ? 20 : N;
            int *idxs = malloc(sizeof(int) * k);
            int found = knn_grid(&grid, used, x_prev, y_prev, k, idxs);

            int best = -1;
            double best_dist = HUGE_VAL;
            for (int t = 0; t < found; ++t) {
                int i = idxs[t];
                if (used[i] || isnan(x[i])) continue;
                double dx = x[i] - x_prev, dy = y[i] - y_prev;
                double dis = dx*dx + dy*dy;
                if (vol[i] < vol_prev) dis += 1.0;
                if (dis < best_dist) { best = i; best_dist = dis; }
            }
            free(idxs);

            if (best < 0) {
                for (int i = 0; i < N; ++i) if (!used[i]) { best = i; break; }
            }
            if (best < 0) break;

            for (int c = 0; c < ncols; ++c)
                dfLoop[(size_t)j * ncols + c] = df[(size_t)best * ncols + c];
            used[best] = 1;
        }

        free_grid(&grid);

        char outpath[1024];
        snprintf(outpath, sizeof(outpath), "%sloop_%s", folName, fname);
        printf("save %s\n", outpath);
        save_txt(outpath, dfLoop + ncols, N - 1, ncols);

        free(df); free(dfLoop); free(x); free(y); free(vol); free(used);
    }
    closedir(dir);
    return 0;
}

/* ------------------------------------------------------------------------- */
/* Main                                                                      */
/* ------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    /* If called with 3 arguments: ./run <capLen> <RadTop> <fname> */
    if (argc == 4) {
        double capLen = atof(argv[1]);
        double RadTop = atof(argv[2]);
        const char *fname = argv[3];
        AdamsBashforthProfile(capLen, RadTop, -1.0, fname,
                              0.0, 0.0, 0.0, 0.0,
                              NULL, NULL, NULL, NULL, NULL, NULL);        
        return 0;
    }

    /* Create the simData directory if it doesn't exist */
    mkdir("simData", 0755);

    /* Initialize global bin arrays (valid = 0) */
    for (int i = 0; i < MAX_BINS; i++) {
        radBinsVol[i].valid = 0;
        angBinsVol[i].valid = 0;
        angBinsMaxRad[i].valid = 0;
        radBinsMaxAng[i].valid = 0;
    }

    /* Build RadTops = logspace(-1, 1, 0) => no profiles for fine bins */
    double log_start = -1.0, log_stop = 1.0;
    int N = 10000;
    for (int b = 0; b < N; ++b) {
        double t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
        double log_val = log_start + t * (log_stop - log_start);
        double RadTop = pow(10.0, log_val);
        
        int i;   // will hold the internal iteration count
        AdamsBashforthProfile(1.0, RadTop, -1.0, NULL,
                              0.001, 0.01,   /* fine bins for consolidated max */
                              0,  0,    /* coarse bins for per‑bin files   */
                              NULL, NULL, NULL, NULL, NULL, &i);

        if (b % 100 == 0) {
            printf("b %d of %d RadTop %.10g i %d\n", b, N, RadTop, i);
        }
    }

    /* Write all consolidated files (only if N > 0) */
    if (N) {
        write_bin_file("simData/ConstAngMaxVol.txt", angBinsVol, maxAngBinVolUsed);
        write_bin_file("simData/ConstRadMaxVol.txt", radBinsVol, maxRadBinVolUsed);
        write_bin_file("simData/ConstAngMaxRad.txt", angBinsMaxRad, maxAngBinRadUsed);
        write_bin_file("simData/ConstRadMaxAng.txt", radBinsMaxAng, maxRadBinAngUsed);
    }

    /* Now run coarse bins for per‑bin files */
    N = 10000;
    log_start = -2.0; log_stop = 2.0;
    for (int b = 0; b < N; ++b) {
        double t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
        double log_val = log_start + t * (log_stop - log_start);
        double RadTop = pow(10.0, log_val);
        
        int i;
        AdamsBashforthProfile(1.0, RadTop, -1.0, NULL,
                              0, 0,   /* fine bins for consolidated max */
                              0.01,  0.1,    /* coarse bins for per‑bin files   */
                              NULL, NULL, NULL, NULL, NULL, &i);

        if (b % 100 == 0) {
            printf("b %d of %d RadTop %.10g i %d\n", b, N, RadTop, i);
        }
    }

    reorder_drop_height_vs_vol("ang");
    reorder_drop_height_vs_vol("rad");

    return 0;
}
