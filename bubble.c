/*
 * bubble.c
 * Ianto Cannon 2025 Feb 13.
 * Functions for calculating the interface shape of bubbles on surfaces.
 * Adapted from code written by Stefan Endres.
 *
 * C port of bubble.py.  No external libraries beyond the C standard library
 * and POSIX dirent are required.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>    /* Added for isspace() */
#include <dirent.h>
#include <float.h>

#define PI 3.14159265358979323846
#define MAX_ITER 1000000
#define PROFILE_NCOLS 11   /* r, -z, psi, dPsi, capLen, RadTop, Volume, area, centroid/Volume, maxRad */
#define BIN_NCOLS    10    /* same as profile but without the trailing maxRad... actually
                              Python prints 11 for the profile file and 10 for the bin files.
                              Keeping that asymmetry faithful. */

/* ------------------------------------------------------------------------- */
/* AdamsBashforthProfile                                                     */
/* ------------------------------------------------------------------------- */
int AdamsBashforthProfile(double capLen, double RadTop,
                          double contactAng /* unused in py */, const char *fname,
                          double angleSave, double radSave,
                          double *outVolume, double *out_r, double *out_z,
                          double *out_centroid, double *out_psi)
{
    /* compute analytical interface shape according to eq 1 of Demirkir2024Langmuir.
       Input the Bond number Bo (=1/capLen^2) and the radius of curvature at bubble top RadTop.
       Returns: volume, contact-patch radius, bubble height, height of centre of mass. */

    double ds = fmin(fmin(1e-4 * fabs(RadTop), 1e-4 * fabs(capLen)), 1e-4);
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

        /* guard r==0 (only at i=0) */
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
                        "%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                        r, -z, psi, dPsi, capLen, RadTop, Volume, area,
                        (Volume != 0.0 ? centroid / Volume : 0.0), maxRad);
            }
        }

        if (angleSave > 0.0 && i > 0) {
            int angBin     = (int)floor((psi      / PI) / angleSave);
            int angBinPrev = (int)floor(((psi-dPsi)/ PI) / angleSave);
            if (angBin != angBinPrev) {
                int b = (angBin > angBinPrev) ? angBin : angBinPrev;
                char angFname[256];
                snprintf(angFname, sizeof(angFname), "simData/ang%05d.txt", b);
                FILE *ang_txt = fopen(angFname, "a");
                if (ang_txt) {
                    fprintf(ang_txt,
                            "%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                            r, -z, psi, dPsi, capLen, RadTop, Volume, area,
                            (Volume != 0.0 ? centroid / Volume : 0.0), maxRad);
                    fclose(ang_txt);
                }
            }
        }

        if (radSave > 0.0 && i > 0) {
            int radBin     = (int)floor(r       / radSave);
            int radBinPrev = (int)floor((r - dr) / radSave);
            if (radBin != radBinPrev) {
                int b = (radBin > radBinPrev) ? radBin : radBinPrev;
                char radFname[256];
                snprintf(radFname, sizeof(radFname), "simData/rad%05d.txt", b);
                FILE *rad_txt = fopen(radFname, "a");
                if (rad_txt) {
                    fprintf(rad_txt,
                            "%.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
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

    printf("saved %s capLen %.6g RadTop%10.6g i %d\n",
           fname ? fname : "(null)", capLen, RadTop, i);

    if (outVolume)  *outVolume  = Volume;
    if (out_r)      *out_r      = r;
    if (out_z)      *out_z      = z;
    if (out_centroid)*out_centroid = (Volume != 0.0 ? centroid / Volume : 0.0);
    if (out_psi)    *out_psi    = psi;
    return 0;
}

/* ------------------------------------------------------------------------- */
/* reorder_drop_height_vs_vol                                                */
/* ------------------------------------------------------------------------- */

/* Tokenise a whitespace-separated line into doubles.
   Returns the number of fields parsed (<= max_cols). */
static int parse_line_doubles(const char *line, double *out, int max_cols)
{
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

/* Load a whitespace-separated numeric file into a row-major matrix.
   Returns 0 on success, sets *out_rows, *out_cols, *out_data (malloc'd). */
static int load_txt(const char *path, double **out_data, int *out_rows, int *out_cols)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    /* First pass: count rows and max columns. */
    int rows = 0, cols = 0;
    char *line = NULL; size_t cap = 0; ssize_t len;
    double *buf = NULL; int buf_cap = 0;

    /* We need two passes; simplest: read into a growable list. */
    while ((len = getline(&line, &cap, f)) != -1) {
        /* count tokens */
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
        /* Re-parse with current row stride = cols */
        {
            double tmp[64];
            int n = parse_line_doubles(line, tmp, (c < 64 ? c : 64));
            /* If c>64 we'd need a bigger tmp; for these inputs c<=11 so it's fine. */
            for (int k = 0; k < n; ++k) buf[(size_t)rows * cols + k] = tmp[k];
            for (int k = n; k < cols; ++k) buf[(size_t)rows * cols + k] = 0.0;
        }
        rows++;
    }
    fclose(f);
    free(line);

    if (rows == 0 || cols == 0) { free(buf); return -1; }
    *out_data = buf; *out_rows = rows; *out_cols = cols;
    return 0;
}

static int save_txt(const char *path, const double *data, int rows, int cols)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            fprintf(f, "%.18e%s", data[(size_t)i * cols + j],
                    (j == cols - 1) ? "\n" : " ");
        }
    }
    fclose(f);
    return 0;
}

/* Brute-force k nearest neighbours.  Writes the indices of the k nearest
   points (to the query point qx,qy) into out_idx, sorted by distance.
   Excluded points (mask==1) are skipped.
   Returns the actual number of neighbours found (<= k, <= N). */
static int knn_bruteforce(const double *x, const double *y, const unsigned char *mask,
                          int N, double qx, double qy, int k, int *out_idx)
{
    /* We do a simple selection: compute all distances, then partial-sort. */
    if (k > N) k = N;
    if (k <= 0) return 0;

    /* Allocate temporary arrays. */
    double *dist = malloc(sizeof(double) * N);
    int   *idx   = malloc(sizeof(int)   * N);
    if (!dist || !idx) { free(dist); free(idx); return 0; }

    int cnt = 0;
    for (int i = 0; i < N; ++i) {
        if (mask[i]) continue;
        if (isnan(x[i]) || isnan(y[i])) continue;
        double dx = x[i] - qx, dy = y[i] - qy;
        dist[cnt] = dx*dx + dy*dy;
        idx[cnt]  = i;
        cnt++;
    }

    /* Partial selection sort to get the first k. */
    int found = (cnt < k) ? cnt : k;
    for (int j = 0; j < found; ++j) {
        int best = j;
        for (int t = j + 1; t < cnt; ++t) if (dist[t] < dist[best]) best = t;
        if (best != j) {
            double td = dist[j]; dist[j] = dist[best]; dist[best] = td;
            int ti = idx[j]; idx[j] = idx[best]; idx[best] = ti;
        }
        out_idx[j] = idx[j];
    }
    free(dist); free(idx);
    return found;
}

int reorder_drop_height_vs_vol(const char *nam)
{
    const char *folName = "simData/";
    DIR *dir = opendir(folName);
    if (!dir) { perror("opendir"); return -1; }

    struct dirent *de;
    while ((de = readdir(dir)) != NULL) {
        const char *fname = de->d_name;

        /* skip if "loop" in name or "txt" not in name or nam not in name */
        if (strstr(fname, "loop")) continue;
        if (!strstr(fname, "txt")) continue;
        if (nam && *nam && !strstr(fname, nam)) continue;

        char path[1024];
        snprintf(path, sizeof(path), "%s%s", folName, fname);

        double *df = NULL; int N = 0, ncols = 0;
        if (load_txt(path, &df, &N, &ncols) != 0) continue;
        if (N < 2) { free(df); continue; }

        /* dfLoop = zeros_like(df); dfLoop[:,4] = 1 */
        double *dfLoop = calloc((size_t)N * ncols, sizeof(double));
        for (int i = 0; i < N; ++i) dfLoop[(size_t)i * ncols + 4] = 1.0;

        /* Feature space:
           x   = df[:,1] / df[:,4]
           y   = cbrt(df[:,6]) / df[:,4]
           vol = df[:,6] / df[:,4]^3
           Python uses NaN-aware division; C produces inf/nan naturally. */
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

        /* dfLoop[0] = df[0]   (Note: Python code has these two lines
           commented out, so the loop starts with all-zero dfLoop except col4=1.
           To match Python exactly we leave row 0 of dfLoop as zeros with col4=1.) */

        for (int j = 1; j < N; ++j) {
            double row1 = dfLoop[(size_t)(j-1) * ncols + 1];
            double row4 = dfLoop[(size_t)(j-1) * ncols + 4];
            double row6 = dfLoop[(size_t)(j-1) * ncols + 6];
            double x_prev   = row1 / row4;
            double y_prev   = cbrt(row6) / row4;
            double vol_prev = row6 / (row4 * row4 * row4);

            int k = (20 < N) ? 20 : N;
            int *idxs = malloc(sizeof(int) * k);
            int found = knn_bruteforce(x, y, used, N, x_prev, y_prev, k, idxs);

            int best = -1;
            double best_dist = HUGE_VAL;
            for (int t = 0; t < found; ++t) {
                int i = idxs[t];
                if (used[i] || isnan(x[i])) continue;
                double dx = x[i] - x_prev, dy = y[i] - y_prev;
                double dis = dx*dx + dy*dy;
                if (vol[i] < vol_prev) dis += 1.0; /* volume penalty */
                if (dis < best_dist) { best = i; best_dist = dis; }
            }
            free(idxs);

            if (best < 0) {
                /* Fallback: first unused index. */
                for (int i = 0; i < N; ++i) if (!used[i]) { best = i; break; }
            }
            if (best < 0) break;

            for (int c = 0; c < ncols; ++c)
                dfLoop[(size_t)j * ncols + c] = df[(size_t)best * ncols + c];
            used[best] = 1;
        }

        char outpath[1024];
        snprintf(outpath, sizeof(outpath), "%sloop_%s", folName, fname);
        printf("save %s\n", outpath);
        save_txt(outpath, dfLoop, N, ncols);

        free(df); free(dfLoop); free(x); free(y); free(vol); free(used);
    }
    closedir(dir);
    return 0;
}
