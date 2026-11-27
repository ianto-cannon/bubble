#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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

/* Per-bin files (angNNNNN.bin, radNNNNN.bin) are headerless streams of records
 * of FILE_COLS native-endian floats: the columns of the enum above, in order,
 * without IDX_CAPLEN. The Z column is stored negated.
 *
 * During the sweep, records are collected as full BinData doubles in .raw files.
 * Reordering sorts them in double precision, writes the .bin file and removes
 * the .raw file. */
#define FILE_COLS 9

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
static void append_record(const char *path, const double *state) {
  BinData rec;
  memcpy(rec, state, sizeof(rec));
  rec[IDX_Z] = -rec[IDX_Z];

  FILE *f = fopen(path, "ab");
  if (!f) { perror(path); exit(EXIT_FAILURE); }
  size_t written = fwrite(rec, sizeof(rec), 1, f);
  if (fclose(f) != 0 || written != 1) { perror(path); exit(EXIT_FAILURE); }
}

static int load_raw(const char *path, BinData **out_data, int *out_rows) {
  FILE *f = fopen(path, "rb");
  if (!f) return -1;

  long size = -1;
  if (fseek(f, 0, SEEK_END) == 0) size = ftell(f);
  if (size <= 0 || size % (long)sizeof(BinData) != 0) {
    fclose(f);
    return -1;
  }
  rewind(f);

  size_t rows = (size_t)size / sizeof(BinData);
  BinData *buf = malloc((size_t)size);
  if (!buf || fread(buf, sizeof(BinData), rows, f) != rows) {
    free(buf);
    fclose(f);
    return -1;
  }

  fclose(f);
  *out_data = buf; *out_rows = (int)rows;
  return 0;
}

/* Writes records as floats, omitting capLen. The data goes to a temporary file
 * that is then renamed to the target, so no partial .bin file is left behind. */
static int save_bin(const char *path, const BinData *data, int rows) {
  char tmp[1100];
  snprintf(tmp, sizeof(tmp), "%s.tmp", path);

  FILE *f = fopen(tmp, "wb");
  if (!f) return -1;

  int ok = 1;
  for (int i = 0; ok && i < rows; ++i) {
    float rec[FILE_COLS];
    int c = 0;
    for (int j = 0; j < 10; ++j)
      if (j != IDX_CAPLEN) rec[c++] = (float)data[i][j];
    ok = fwrite(rec, sizeof(rec), 1, f) == 1;
  }

  if (fclose(f) != 0 || !ok || rename(tmp, path) != 0) {
    remove(tmp);
    return -1;
  }
  return 0;
}

/* Saves the final .bin file; the .raw file is removed only if that succeeds. */
static void finish_file(const char *rawpath, const char *binpath,
                        const BinData *data, int rows) {
  if (save_bin(binpath, data, rows) != 0)
    fprintf(stderr, "Error writing %s\n", binpath);
  else if (remove(rawpath) != 0)
    perror(rawpath);
  else
    printf("saved %s\n", binpath);
}

/* True for names of the form <prefix>NNNNN<ext> */
static int is_data_file(const char *fname, const char *prefix, const char *ext) {
  size_t plen = strlen(prefix);
  if (strncmp(fname, prefix, plen) != 0) return 0;
  const char *p = fname + plen;
  size_t digits = strspn(p, "0123456789");
  return digits > 0 && strcmp(p + digits, ext) == 0;
}

/* Clear per-bin files left by an earlier run, so old and new data never mix. */
static void remove_old_data_files(void) {
  DIR *dir = opendir(SIMDATA_DIR);
  if (!dir) { perror(SIMDATA_DIR); exit(EXIT_FAILURE); }

  struct dirent *de;
  while ((de = readdir(dir)) != NULL) {
    const char *name = de->d_name;
    if (!is_data_file(name, "ang", ".raw") && !is_data_file(name, "rad", ".raw") &&
        !is_data_file(name, "ang", ".bin") && !is_data_file(name, "rad", ".bin"))
      continue;
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", SIMDATA_DIR, de->d_name);
    if (remove(path) != 0) { perror(path); exit(EXIT_FAILURE); }
  }
  closedir(dir);
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
      if ((i % 1000 == 0) || (state[IDX_DPSI] > 1e-2)) {
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
        snprintf(angFname, sizeof(angFname), "%s/ang%05d.raw", SIMDATA_DIR, b);
        append_record(angFname, state);
      }
    }

    /* Large radius bins (per-bin files) */
    if (radSaveLarge > 0.0 && i > 0) {
      int radBinLarge   = (int)floor(state[IDX_R] / radSaveLarge);
      int radBinLargePrev = (int)floor((state[IDX_R] - dr) / radSaveLarge);
      if (radBinLarge != radBinLargePrev) {
        int b = (radBinLarge > radBinLargePrev) ? radBinLarge : radBinLargePrev;
        char radFname[256];
        snprintf(radFname, sizeof(radFname), "%s/rad%05d.raw", SIMDATA_DIR, b);
        append_record(radFname, state);
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
  fprintf(fp, "# r -z psi dPsi capLen RadTop Volume area centroid maxRad\n");
  for (int b = 0; b <= max_bin_used; b++) {
    if (bins[b][IDX_VOLUME] >= 0.0) {  // sentinel check
      for (int j = 0; j < 10; ++j) {
        double val = bins[b][j];
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
/* Round to 10 significant digits. The greedy walk below penalises any drop in
 * volume, so differences in the last bits between near-identical points would
 * change the ordering. Used for comparisons only; stored data is not rounded. */
static double round10(double v)
{
  char text[32];
  snprintf(text, sizeof(text), "%.9e", v);
  return strtod(text, NULL);
}

static void reorder_file(const char *rawpath, const char *binpath)
{
  BinData *df = NULL; int N = 0;
  if (load_raw(rawpath, &df, &N) != 0) {
    fprintf(stderr, "Skipping unreadable file %s\n", rawpath);
    return;
  }
  if (N < 2) { /* nothing to reorder */
    finish_file(rawpath, binpath, (const BinData *)df, N);
    free(df);
    return;
  }

  BinData *dfLoop = calloc((size_t)N, sizeof(BinData));
  if (!dfLoop) { fprintf(stderr, "Memory allocation failed\n"); exit(EXIT_FAILURE); }

  double *x   = malloc(sizeof(double) * N);
  double *y   = malloc(sizeof(double) * N);
  double *vol = malloc(sizeof(double) * N);
  for (int i = 0; i < N; ++i) {
    double row1 = round10(df[i][IDX_Z]);
    double row4 = round10(df[i][IDX_CAPLEN]);
    double row6 = round10(df[i][IDX_VOLUME]);
    x[i]   = row1 / row4;
    y[i]   = cbrt(row6) / row4;
    vol[i] = row6 / (row4 * row4 * row4);
  }

  unsigned char *used = calloc(N, sizeof(unsigned char));

  Grid grid;
  build_grid(&grid, x, y, N);

  /* The walk starts from the origin. */
  double x_prev = 0.0, y_prev = 0.0, vol_prev = 0.0;

  for (int j = 0; j < N; ++j) {
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

    memcpy(dfLoop[j], df[best], sizeof(BinData));
    used[best] = 1;
    x_prev = x[best];
    y_prev = y[best];
    vol_prev = vol[best];
  }

  free_grid(&grid);

  finish_file(rawpath, binpath, (const BinData *)dfLoop, N);

  free(df); free(dfLoop); free(x); free(y); free(vol); free(used);
}

/* Reorders every <prefix>NNNNN.raw file in SIMDATA_DIR into <prefix>NNNNN.bin. */
int reorder_drop_height_vs_vol(const char *prefix)
{
  DIR *dir = opendir(SIMDATA_DIR);
  if (!dir) { perror("opendir"); return -1; }

  /* Collect the names first: replacing files while readdir is still walking
   * the directory could return the same file twice. */
  char (*names)[256] = NULL;
  size_t count = 0, cap = 0;
  struct dirent *de;
  while ((de = readdir(dir)) != NULL) {
    if (!is_data_file(de->d_name, prefix, ".raw")) continue;
    if (count == cap) {
      cap = cap ? 2 * cap : 64;
      char (*grown)[256] = realloc(names, cap * sizeof(*names));
      if (!grown) { free(names); closedir(dir); return -1; }
      names = grown;
    }
    snprintf(names[count++], sizeof(*names), "%.255s", de->d_name);
  }
  closedir(dir);

  for (size_t n = 0; n < count; ++n) {
    char rawpath[1024], binpath[1024];
    int stem = (int)(strlen(names[n]) - strlen(".raw"));
    snprintf(rawpath, sizeof(rawpath), "%s/%s", SIMDATA_DIR, names[n]);
    snprintf(binpath, sizeof(binpath), "%s/%.*s.bin", SIMDATA_DIR, stem, names[n]);
    reorder_file(rawpath, binpath);
  }
  free(names);
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
  remove_old_data_files();

  const double angleSave_fine = 0.001;
  const double radSave_fine   = 0.0001;
  const double angleSave_coarse = 0.02;
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

  int N = 10000;
  for (int b = 0; b < N; ++b) {
    double t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
    double log_start = -1.0, log_stop = 0.0;
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
      printf("b %d of %d RadTop %.10g iterations %d\n", b, N, RadTop, i);
    
    t = (N == 1) ? 0.0 : (double)b / (double)(N - 1);
    log_start = -3.0, log_stop = 3.0;
    log_val = log_start + t * (log_stop - log_start);
    RadTop = pow(10.0, log_val);
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
      printf("b %d of %d RadTop %.10g iterations %d\n", b, N, RadTop, i);
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
