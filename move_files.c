#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>


int isDirectoryEmpty(const char *path) {
    DIR *dir = opendir(path);

    if (dir == NULL) {
        perror("Error opening directory");
        exit(EXIT_FAILURE);
    }

    struct dirent *entry;
    int isEmpty = 1;  // Assume directory is empty until proven otherwise

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
            // Directory is not empty
            isEmpty = 0;
            break;
        }
    }

    closedir(dir);

    return isEmpty;
}


void moveFilesAndRemoveEmptyDirs(const char *directory) {
    DIR *dir = opendir(directory);

    if (dir == NULL) {
        perror("Error al abrir el directorio");
        exit(EXIT_FAILURE);
    }

    struct dirent *entry;

    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_DIR) {
            if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
                char subdirPath[1024];
                snprintf(subdirPath, sizeof(subdirPath), "%s/%s", directory, entry->d_name);

                // Mover archivos al directorio anterior
                DIR *subdir = opendir(subdirPath);
                struct dirent *subentry;

                while ((subentry = readdir(subdir)) != NULL) {
                    if (subentry->d_type == DT_REG) {
                        char filePath[1024];
                        snprintf(filePath, sizeof(filePath), "%s/%s", subdirPath, subentry->d_name);
                        
                        char destPath[1024];
                        snprintf(destPath, sizeof(destPath), "%s/%s", directory, subentry->d_name);

                        if (rename(filePath, destPath) != 0) {
                            perror("Error al mover el archivo");
                            exit(EXIT_FAILURE);
                        }
                    }
                }

                closedir(subdir);
                if (isDirectoryEmpty(subdirPath) == 1){
                    // Eliminar el directorio si está vacío
                    if (rmdir(subdirPath) != 0) {
                        perror("Error al borrar el directorio");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }

    closedir(dir);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s directorio\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    moveFilesAndRemoveEmptyDirs(argv[1]);

    return 0;
}
