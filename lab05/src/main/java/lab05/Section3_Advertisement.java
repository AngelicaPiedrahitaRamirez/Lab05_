package lab05;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.File;
import java.util.Random;

/**
 * Section 3 – Advertisement Dataset
 * Replicates the R pipeline from the lab using Weka J48.
 */
public class Section3_Advertisement {

    public static void run() throws Exception {
        System.out.println("==========================================================");
        System.out.println("  SECTION 3 - Advertisement Decision Tree (Weka / J48)");
        System.out.println("==========================================================\n");

        // ---------------------------------------------------------------
        // 3.2  Load the dataset
        // ---------------------------------------------------------------
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/resources/Advertisement.csv"));
        Instances data = loader.getDataSet();

        System.out.println("Dataset loaded. Total instances: " + data.numInstances());
        System.out.println("Attributes: " + data.numAttributes());

        // ---------------------------------------------------------------
        // 3.3  Convertir Purchased a nominal (factor) - igual que en R:
        //      dataset$Purchased <- factor(dataset$Purchased, levels = c(0,1))
        // ---------------------------------------------------------------
        NumericToNominal toNominal = new NumericToNominal();
        toNominal.setAttributeIndices("last"); // Purchased es la ultima columna
        toNominal.setInputFormat(data);
        data = Filter.useFilter(data, toNominal);

        // Remover User.ID (col 1) y Gender (col 2) - solo usar Age y EstimatedSalary
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("1,2");
        removeFilter.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeFilter);

        // Purchased es ahora la ultima columna -> clase
        filteredData.setClassIndex(filteredData.numAttributes() - 1);

        System.out.println("Features used: Age, EstimatedSalary");
        System.out.println("Class: Purchased (0=No compro, 1=Compro)\n");

        // ---------------------------------------------------------------
        // 3.3  Split 75% train / 25% test  (seed=42 igual que R)
        // ---------------------------------------------------------------
        filteredData.randomize(new Random(42));
        filteredData.stratify(4);

        Instances trainSet = filteredData.trainCV(4, 0, new Random(42));
        Instances testSet  = filteredData.testCV(4, 0);

        System.out.println("Training set size: " + trainSet.numInstances());
        System.out.println("Test set size:     " + testSet.numInstances() + "\n");

        // ---------------------------------------------------------------
        // 3.4  Feature scaling - equivalente a scale() en R
        // ---------------------------------------------------------------
        Standardize scaler = new Standardize();
        scaler.setInputFormat(trainSet);
        Instances trainScaled = Filter.useFilter(trainSet, scaler);
        Instances testScaled  = Filter.useFilter(testSet,  scaler);

        System.out.println("Feature scaling aplicado (media=0, desv=1)\n");

        // ---------------------------------------------------------------
        // 3.5  Entrenar el clasificador J48
        //      Equivalente a: rpart(Purchased ~ Age + EstimatedSalary, ...)
        // ---------------------------------------------------------------
        J48 classifier = new J48();
        classifier.buildClassifier(trainScaled);

        // ---------------------------------------------------------------
        // 3.6  Imprimir el arbol (reemplaza rpart.plot en R)
        // ---------------------------------------------------------------
        System.out.println("--- Estructura del Arbol de Decision ---");
        System.out.println(classifier.toString());

        // ---------------------------------------------------------------
        // 3.7  Evaluacion: Matriz de confusion + Accuracy
        // ---------------------------------------------------------------
        Evaluation eval = new Evaluation(trainScaled);
        eval.evaluateModel(classifier, testScaled);

        System.out.println("--- Matriz de Confusion ---");
        System.out.println(eval.toMatrixString());

        double[][] cm = eval.confusionMatrix();
        System.out.println("Confusion Matrix (filas=Real, columnas=Predicho):");
        System.out.printf("              Predicho 0   Predicho 1%n");
        System.out.printf("  Real 0    :    %6.0f        %6.0f%n", cm[0][0], cm[0][1]);
        System.out.printf("  Real 1    :    %6.0f        %6.0f%n", cm[1][0], cm[1][1]);

        double accuracy = eval.pctCorrect() / 100.0;
        System.out.printf("%nModel Accuracy: %.2f (%.1f%%)%n", accuracy, eval.pctCorrect());

        System.out.println("\n--- Metricas Adicionales ---");
        System.out.printf("Precision (clase 1): %.2f%n", eval.precision(1));
        System.out.printf("Recall    (clase 1): %.2f%n", eval.recall(1));
        System.out.printf("F1-Score  (clase 1): %.2f%n", eval.fMeasure(1));
    }
}