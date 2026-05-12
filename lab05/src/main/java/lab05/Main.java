package lab05;

/**
 * Lab 05 – Decision Trees in Java using Weka
 *
 * Runs both sections sequentially:
 *   - Section 2: Play Tennis dataset
 *   - Section 3: Advertisement dataset (autonomous work)
 */
public class Main {

    public static void main(String[] args) {
        try {
            Section2_PlayTennis.run();
            Section3_Advertisement.run();
        } catch (Exception e) {
            System.err.println("Error during execution: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
