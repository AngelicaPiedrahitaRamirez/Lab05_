# Lab 05 - Decision Trees in Java (Weka)

## Requisitos
- Java JDK 17 → https://adoptium.net/temurin/releases/?version=17
- Maven (mvnd) → https://maven.apache.org/download.cgi

## Comandos

### Compilar
```bash
mvnd package
```

### Ejecutar
```bash
java --add-opens java.base/java.lang=ALL-UNNAMED -jar target\decision-trees-1.0-SNAPSHOT-shaded.jar
```

## Datasets requeridos
Deben estar en `src/main/resources/`:
- `Advertisement.csv`
- `play_tennis_dataset.csv`
