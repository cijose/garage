#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
float squared_l2_distance(int d, const float *x1, const float *x2) {
    float dist = 0;
    for (int i = 0; i < d; i++) {
        dist += pow(x1[i] - x2[i], 2);
    }
    return dist;
}
inline float min(float a, float b) { return a < b ?: b; }
void initialize_kmeanspp(int n, int d, int k, const float *points,
                         float *centers) {
    float *energies = (float *)malloc(sizeof(float) * n);
    bool *choosen = (bool *)malloc(sizeof(bool) * n);
    memset(choosen, 0, sizeof(bool) * n);
    int first_point = rand() % n;
    choosen[first_point] = true;
    memcpy(centers, points + first_point * d, sizeof(float) * d);
    float total_energy = 0.f;
    for (int i = 0; i < n; i++) {
        energies[i] = squared_l2_distance(d, points + i * d, centers);
        total_energy += energies[i];
    }
    for (int i = 1; i < k; i++) {
        float *centersi = centers + i * d;
        int choosen_index = -1;
        do {
            float rv = ((float)rand() / (RAND_MAX));
            float pdf = 0.0;
            for (int j = 0; j < n; j++) {
                pdf += (energies[j] / total_energy);
                if (pdf >= rv) {
                    choosen_index = j;
                    break;
                }
            }
        } while (choosen[choosen_index] == true);
        memcpy(centersi, points + choosen_index * d, sizeof(float) * d);
        total_energy = 0.f;
        for (int j = 0; j < n; j++) {
            float dist = squared_l2_distance(d, points + j * d, centersi);
            energies[i] = min(dist, energies[i]);
            total_energy += energies[i];
        }
        choosen[choosen_index] = true;
    }
    free(energies);
    free(choosen);
}
