#ifndef BUBBLE_H
#define BUBBLE_H

int AdamsBashforthProfile(double capLen, double RadTop,
                          double contactAng, const char *fname,
                          double angleSave, double radSave,
                          double *outVolume, double *out_r,
                          double *out_z, double *out_centroid,
                          double *out_psi);

int reorder_drop_height_vs_vol(const char *nam);

#endif /* BUBBLE_H */
