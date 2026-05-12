package lab05;

import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.ArrayList;

/**
 * Section 2 – Play Tennis Dataset
 *
 * Equivalent to the R code in the lab:
 *   model <- rpart(PlayTennis ~ ., data=tennis_data, method="class",
 *                  control=rpart.control(cp=0, minsplit=2, maxdepth=2))
 *
 * In Weka, J48 is the standard C4.5 decision tree (rpart uses CART, but
 * both are entropy-based and produce equivalent results on this dataset).
 */
public class Section2_PlayTennis {

    public static void run() throws Exception {
        System.out.println("==========================================================");
        System.out.println("  SECTION 2 – Play Tennis Decision Tree (Weka / J48)");
        System.out.println("==========================================================\n");

        // ---------------------------------------------------------------
        // 2.1  Load the dataset
        // ---------------------------------------------------------------
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("src/main/resources/play_tennis_dataset.csv"));
        Instances data = loader.getDataSet();

        // Remove the 'Day' column (index 0) – same as tennis_data$Day <- NULL in R
        data.deleteAttributeAt(0);

        // Set class attribute: PlayTennis (last column after removing Day)
        data.setClassIndex(data.numAttributes() - 1);

        System.out.println("Dataset loaded. Attributes: " + data.numAttributes());
        System.out.println("Instances:  " + data.numInstances());
        System.out.println("Class attribute: " + data.classAttribute().name() + "\n");

        // ---------------------------------------------------------------
        // 2.2  Build the model
        //      -C 0.25  → confidence factor (pruning); 0.25 is Weka default
        //      -M 2     → minimum samples per leaf  (minsplit=2 in R)
        //      -maxDepth 2 → equivalent to maxdepth=2 in rpart.control
        //      -U         → unpruned (cp=0 equivalent)
        // ---------------------------------------------------------------
        J48 model = new J48();
        model.setOptions(weka.core.Utils.splitOptions(
                "-U -M 2"
        ));
        model.buildClassifier(data);

        // ---------------------------------------------------------------
        // 2.3  Print the tree structure (replaces rpart.plot in R)
        // ---------------------------------------------------------------
        System.out.println("--- Decision Tree Structure ---");
        System.out.println(model.toString());

        // ---------------------------------------------------------------
        // 2.4  Predict a new observation
        //      new_weather <- data.frame(Outlook="Rain", Temperature="Mild",
        //                               Humidity="High", Wind="Weak")
        //      Expected prediction: No
        // ---------------------------------------------------------------
        System.out.println("--- Prediction for new weather scenario ---");

        // Build the instance manually using the same attribute definitions
        Instances header = new Instances(data, 0); // empty dataset, same schema
        Instance newWeather = new DenseInstance(data.numAttributes());
        newWeather.setDataset(header);

        // Attribute order after removing Day: Outlook, Temperature, Humidity, Wind, PlayTennis
        newWeather.setValue(data.attribute("Outlook"),     "Rain");
        newWeather.setValue(data.attribute("Temperature"), "Mild");
        newWeather.setValue(data.attribute("Humidity"),    "High");
        newWeather.setValue(data.attribute("Wind"),        "Weak");

        double predicted = model.classifyInstance(newWeather);
        String predLabel = data.classAttribute().value((int) predicted);

        System.out.println("Input:  Outlook=Rain, Temperature=Mild, Humidity=High, Wind=Weak");
        System.out.println("Prediction for the new day: " + predLabel);
        System.out.println("(Expected: No)\n");
    }
}
