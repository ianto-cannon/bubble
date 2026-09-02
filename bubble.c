#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#define PI 3.14159265358979323846
#define MAX_ITER 1000000
#define SENTINEL_VOLUME (-1.0)

/* Variable holding the simData directory name */
static const char SIMDATA_DIR[] = "simData";

/* Indices for double[10] array (BinData) */
enum {
  IDX_R = 0,
  IDX_Z = 1,
  IDX_PSI = 2,
  IDX_DPSI = 3,
  IDX_CAPLEN = 4,
  IDX_RADTOP = 5,
  IDX_VOLUME = 6,
  IDX_AREA = 7,
  IDX_CENTROID = 8,
  IDX_MAXRAD = 9
};

typedef double BinData[10];

/* ------------------------------------------------------------------------- */
/* Standard C Line Reader (Replaces POSIX getline)                            */
/* ------------------------------------------------------------------------- */
static char *read_line(FILE *f) {
    size_t cap = 128;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    size_t len = 0;
    int c;
    
    while ((c = fgetc(f)) != EOF) {
        if (len + 1 >= cap) {
            cap *= 2;
            char *new_buf = realloc(buf, cap);
            if (!new_buf) { free(buf); return NULL; }
            buf = new_buf;
        }
        buf[len++] = (char)c;
        if (c == '\n') break;
    }
    
    if (len == 0 && c == EOF) {
        free(buf);
        return NULL;
    }
    buf[len] = '\0';
    return buf;
}

/* ------------------------------------------------------------------------- */
/* Grid-based nearest neighbour routines                                      */
/* ------------------------------------------------------------------------- */
typedef struct {
  double dist;
  int idx;
} Neighbor;

typedef struct {
  const double *x, *y;
  double min_x, min_y, max_x, max_y;
  double cell_size;
  int nx, ny;
  int *head;
  int *next;
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
/* File utilities                                                             */
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

/* Two-pass pure C loader (Robust memory handling) */
static int load_txt(const char *path, double **out_data, int *out_rows, int *out_cols) {
  FILE *f = fopen(path, "r");
  if (!f) return -1;

  int rows = 0, cols = 0;
  char *line = NULL;

  /* Pass 1: Count rows and max cols */
  while ((line = read_line(f)) != NULL) {
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
    if (c > 0) {
      rows++;
      if (c > cols) cols = c;
    }
    free(line);
  }

  if (rows == 0 || cols == 0) {
    fclose(f);
    return -1;
  }

  double *buf = malloc((size_t)rows * cols * sizeof(double));
  if (!buf) {
    fclose(f);
    return -1;
  }
  for (size_t i = 0; i < (size_t)rows * cols; ++i) buf[i] = 0.0;

  /* Pass 2: Read data */
  rewind(f);
  int r = 0;
  while ((line = read_line(f)) != NULL && r < rows) {
    double tmp[64];
    int n = parse_line_doubles(line, tmp, (cols < 64 ? cols : 64));
    for (int k = 0; k < n; ++k) {
      buf[(size_t)r * cols + k] = tmp[k];
    }
    free(line);
    r++;
  }

  fclose(f);
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
/* Adams-Bashforth profile integration                                        */
/* ------------------------------------------------------------------------- */
int AdamsBashforthProfile(double capLen, double RadTop,
              const char *fname,
              double angleSave, double radSave,
              double angleSaveLarge, double radSaveLarge,
              BinData *angBinsMaxVol, int nAngBins,
              BinData *angBinsMaxRad, int nAngBinsMaxRad,
              BinData *angBinsMaxWid, int nAngBinsMaxWid,
              BinData *radBinsMaxVol, int nRadBins,
              BinData *radBinsMaxAng, int nRadBinsMaxAng,
              BinData *radBinsMaxWid, int nRadBinsMaxWid,
              int *out_i)
{
  int i;
  double ds = fmin(fmin(1e-5 * fabs(RadTop), 1e-5 * fabs(capLen)), 1e-5);
  double state[10];
  memset(state, 0, sizeof(state));
  state[IDX_CAPLEN] = capLen;
  state[IDX_RADTOP] = RadTop;

  double drPrev = 0.0, dzPrev = 0.0, dPsiPrev = 0.0;
  double centroid_sum = 0.0;

  FILE *adams_txt = NULL;
  if (fname) adams_txt = fopen(fname, "w");

  for (i = 0; i < MAX_ITER; ++i) {
    double dr = ds * cos(state[IDX_PSI]);
    double dz = ds * sin(state[IDX_PSI]);
    if (i == 0) { drPrev = dr; dzPrev = dz; }

    state[IDX_R] += 1.5 * dr - 0.5 * drPrev;
    state[IDX_Z] += 1.5 * dz - 0.5 * dzPrev;
    drPrev = dr;
    dzPrev = dz;

    double sinpsi_over_r = (state[IDX_R] > 0.0) ? (sin(state[IDX_PSI]) / state[IDX_R]) : 0.0;
    state[IDX_DPSI] = ds * (2.0 / RadTop - state[IDX_Z] / (capLen * capLen) - sinpsi_over_r);
    state[IDX_PSI] += 1.5 * state[IDX_DPSI] - 0.5 * dPsiPrev;

    state[IDX_VOLUME]  += PI * state[IDX_R] * state[IDX_R] * dz;
    centroid_sum       += state[IDX_Z] * PI * state[IDX_R] * state[IDX_R] * dz;
    state[IDX_AREA]    += 2.0 * PI * state[IDX_R] * ds;
    if (state[IDX_R] > state[IDX_MAXRAD]) state[IDX_MAXRAD] = state[IDX_R];

    state[IDX_CENTROID] = (state[IDX_VOLUME] != 0.0 ? centroid_sum / state[IDX_VOLUME] : 0.0);

    if (adams_txt) {
      if ((i % 100 == 0) || (state[IDX_DPSI] > 1e-2)) {
        for (int j = 0; j < 10; ++j) {
          double val = state[j];
          if (j == IDX_Z) val = -val;
          fprintf(adams_txt, " %.10g", val);
        }
        fprintf(adams_txt, "\n");
      }
    }

    /* Fine angle bins */
    if (angleSave > 0.0 && i > 0) {
      int b = (int)floor(state[IDX_PSI] / PI / angleSave);
      if (b >= 0 && b < nAngBins) {
        if (angBinsMaxVol[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_VOLUME] > angBinsMaxVol[b][IDX_VOLUME])
          memcpy(angBinsMaxVol[b], state, sizeof(state));
        if (angBinsMaxRad[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_R] > angBinsMaxRad[b][IDX_R])
          memcpy(angBinsMaxRad[b], state, sizeof(state));
        if (angBinsMaxWid[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_MAXRAD] > angBinsMaxWid[b][IDX_MAXRAD])
          memcpy(angBinsMaxWid[b], state, sizeof(state));
      }
    }

    /* Fine radius bins */
    if (radSave > 0.0 && i > 0) {
      int b = (int)floor(state[IDX_R] / radSave);
      if (b >= 0 && b < nRadBins) {
        if (radBinsMaxVol[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_VOLUME] > radBinsMaxVol[b][IDX_VOLUME])
          memcpy(radBinsMaxVol[b], state, sizeof(state));
        if (radBinsMaxAng[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_PSI] > radBinsMaxAng[b][IDX_PSI])
          memcpy(radBinsMaxAng[b], state, sizeof(state));
        if (radBinsMaxWid[b][IDX_VOLUME] < SENTINEL_VOLUME || state[IDX_MAXRAD] > radBinsMaxWid[b][IDX_MAXRAD])
          memcpy(radBinsMaxWid[b], state, sizeof(state));
      }
    }

    /* Large angle bins (per-bin files) */
    if (angleSaveLarge > 0.0 && i > 0) {
      int angBinLarge   = (int)floor((state[IDX_PSI]    / PI) / angleSaveLarge);
      int angBinLargePrev = (int)floor(((state[IDX_PSI] - state[IDX_DPSI]) / PI) / angleSaveLarge);
      if (angBinLarge != angBinLargePrev) {
        int b = (angBinLarge > angBinLargePrev) ? angBinLarge : angBinLargePrev;
        char angFname[256];
        snprintf(angFname, sizeof(angFname), "%s/ang%05d.txt", SIMDATA_DIR, b);
        FILE *ang_txt = fopen(angFname, "a");
        if (ang_txt) {
          for (int j = 0; j < 10; ++j) {
            double val = state[j];
            if (j == IDX_Z) val = -val;
            fprintf(ang_txt, " %.10g", val);
          }
          fprintf(ang_txt, "\n");
          fclose(ang_txt);
        }
      }
    }

    /* Large radius bins (per-bin files) */
    if (radSaveLarge > 0.0 && i > 0) {
      int radBinLarge   = (int)floor(state[IDX_R] / radSaveLarge);
      int radBinLargePrev = (int)floor((state[IDX_R] - dr) / radSaveLarge);
      if (radBinLarge != radBinLargePrev) {
        int b = (radBinLarge > radBinLargePrev) ? radBinLarge : radBinLargePrev;
        char radFname[256];
        snprintf(radFname, sizeof(radFname), "%s/rad%05d.txt", SIMDATA_DIR, b);
        FILE *rad_txt = fopen(radFname, "a");
        if (rad_txt) {
          for (int j = 0; j < 10; ++j) {
            double val = state[j];
            if (j == IDX_Z) val = -val;
            fprintf(rad_txt, " %.10g", val);
          }
          fprintf(rad_txt, "\n");
          fclose(rad_txt);
        }
      }
    }

    if (state[IDX_PSI] < 0.0) break;
    if (state[IDX_PSI] > PI)  break;
    if (state[IDX_DPSI] > 0.0 && dPsiPrev < 0.0) break;
    dPsiPrev = state[IDX_DPSI];
  }

  if (adams_txt) fclose(adams_txt);

  if (out_i) *out_i = i;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* write_bin_file (uses loop over data)                                       */
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
    if (bins[b][IDX_VOLUME] >= 0.0) {  // sentinel check
      fprintf(fp, "%d", b);
      for (int j = 0; j < 10; ++j) {
        double val = bins[b][j];
        if (j == IDX_Z) val = -val;
        fprintf(fp, " %.10g", val);
      }
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

/* ------------------------------------------------------------------------- */
/* reorder_drop_height_vs_vol (Restored POSIX Directory Listing)              */
/* ------------------------------------------------------------------------- */
int reorder_drop_height_vs_vol(const char *nam)
{
  char folName[1024];
  snprintf(folName, sizeof(folName), "%s/", SIMDATA_DIR);
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
/* Main                                                                       */
/* ------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
  if (argc == 4) {
    double capLen = atof(argv[1]);
    double RadTop = atof(argv[2]);
    const char *fname = argv[3];
    AdamsBashforthProfile(capLen, RadTop, fname,
                0.0, 0.0, 0.0, 0.0,
                NULL, 0, NULL, 0, NULL, 0,
                NULL, 0, NULL, 0, NULL, 0,
                NULL);
    return 0;
  }

  mkdir(SIMDATA_DIR, 0755);

  const double angleSave_fine = 0.001;
  const double radSave_fine   = 0.01;
  const double angleSave_coarse = 0.01;
  const double radSave_coarse   = 0.1;

  int nAng = (angleSave_fine > 0) ? (int)ceil(1.0 / angleSave_fine) + 1 : 0;
  int nRad = (radSave_fine   > 0) ? (int)ceil(4.0 / radSave_fine)   + 1 : 0;

  BinData *angBinsMaxVol = malloc(nAng * sizeof(BinData));
  BinData *angBinsMaxRad = malloc(nAng * sizeof(BinData));
  BinData *angBinsMaxWid = malloc(nAng * sizeof(BinData));
  BinData *radBinsMaxVol = malloc(nRad * sizeof(BinData));
  BinData *radBinsMaxAng = malloc(nRad * sizeof(BinData));
  BinData *radBinsMaxWid = malloc(nRad * sizeof(BinData));

  if ((nAng > 0 && (!angBinsMaxVol || !angBinsMaxRad)) ||
    (nRad > 0 && (!radBinsMaxVol || !radBinsMaxAng))) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  // Initialize sentinel volumes to -1
  for (int i = 0; i < nAng; ++i) {
    angBinsMaxVol[i][IDX_VOLUME]  = SENTINEL_VOLUME;
    angBinsMaxRad[i][IDX_VOLUME] = SENTINEL_VOLUME;
    angBinsMaxWid[i][IDX_VOLUME] = SENTINEL_VOLUME;
  }
  for (int i = 0; i < nRad; ++i) {
    radBinsMaxVol[i][IDX_VOLUME]  = SENTINEL_VOLUME;
    radBinsMaxAng[i][IDX_VOLUME] = SENTINEL_VOLUME;
    radBinsMaxWid[i][IDX_VOLUME] = SENTINEL_VOLUME;
  }

  int N = 1000;
  double log_start = -2.0, log_stop = 2.0;
  for (int b = 0; b < N; ++b) {
    double t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
    double log_val = log_start + t * (log_stop - log_start);
    double RadTop = pow(10.0, log_val);

    int i;
    AdamsBashforthProfile(1.0, RadTop, NULL,
                angleSave_fine, radSave_fine,
                angleSave_coarse, radSave_coarse,
                angBinsMaxVol, nAng,
                angBinsMaxRad, nAng,
                angBinsMaxWid, nAng,
                radBinsMaxVol, nRad,
                radBinsMaxAng, nRad,
                radBinsMaxWid, nRad,
                &i);
    if (b % 100 == 0)
      printf("b %d of %d RadTop %.10g i %d\n", b, N, RadTop, i);
  }

  char outpath[1024];
  snprintf(outpath, sizeof(outpath), "%s/BinAngMaxVol.txt", SIMDATA_DIR);
  write_bin_file(outpath, angBinsMaxVol, nAng - 1);
  snprintf(outpath, sizeof(outpath), "%s/BinAngMaxRad.txt", SIMDATA_DIR);
  write_bin_file(outpath, angBinsMaxRad, nAng - 1);
  snprintf(outpath, sizeof(outpath), "%s/BinAngMaxWid.txt", SIMDATA_DIR);
  write_bin_file(outpath, angBinsMaxWid, nAng - 1);
  snprintf(outpath, sizeof(outpath), "%s/BinRadMaxVol.txt", SIMDATA_DIR);
  write_bin_file(outpath, radBinsMaxVol, nRad - 1);
  snprintf(outpath, sizeof(outpath), "%s/BinRadMaxAng.txt", SIMDATA_DIR);
  write_bin_file(outpath, radBinsMaxAng, nRad - 1);
  snprintf(outpath, sizeof(outpath), "%s/BinRadMaxWid.txt", SIMDATA_DIR);
  write_bin_file(outpath, radBinsMaxWid, nRad - 1);

  free(angBinsMaxVol);
  free(angBinsMaxRad);
  free(angBinsMaxWid);
  free(radBinsMaxVol);
  free(radBinsMaxAng);
  free(radBinsMaxWid);

  reorder_drop_height_vs_vol("ang");
  reorder_drop_height_vs_vol("rad");

  return 0;
}
