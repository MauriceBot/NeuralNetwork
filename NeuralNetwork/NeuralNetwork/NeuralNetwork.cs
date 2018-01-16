using System;
using System.Threading;

namespace NeuralNetwork
{
    public class NeuralNetwork
    {
        private const int INPUTNODES = 784;
        private const int HIDDENNODES = 100;
        private const int OUTPUTNODES = 10;
        private const double LEARNINGRATE = 0.3;

        private int inputnodes;      // The amount of inputs.
        private int hiddennodes;     // The amount of nodes in the hidden layer.
        private int outputnodes;     // The amount of outputs.
        private double learningrate; // The learning rate.

        Matrix weightInputHidden = new Matrix(0, 0);    // Weights between the input layer and the hidden layer.
        Matrix weightHiddenOutput = new Matrix(0, 0);   // Weights between the hidden layer and the outputs layer.

        /// <summary>
        /// Constructor for a general-purpose neural network.
        /// </summary>
        /// <param name="inputnodes"> Amount of inputs. </param>
        /// <param name="hiddennodes"> Amount of nodes in the hidden layer. </param>
        /// <param name="outputnodes"> Amount of outputs. </param>
        /// <param name="learningrate"> The learning rate. </param>
        public NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate)
        {
            this.inputnodes = inputnodes;
            this.hiddennodes = hiddennodes;
            this.outputnodes = outputnodes;
            this.learningrate = learningrate;

            // The weights are given random values between [-0.5,0.5] at the beginning.
            weightInputHidden = RandomizeWeights(inputnodes, hiddennodes);
            weightHiddenOutput = RandomizeWeights(hiddennodes, outputnodes);
        }

        /// <summary>
        /// Query the network for an answer.
        /// </summary>
        /// <param name="inputs"> Inputs for the network. </param>
        /// <returns> The output matrix. </returns>
        public Matrix Query(Matrix inputs)
        {
            // Inputs going into the hidden layer.
            Matrix hidden_inputs = Matrix.DotProduct(inputs, weightInputHidden);
            // Inputs leaving the hidden layer sigmoided.
            Matrix hidden_outputs = hidden_inputs.Sigmoid();

            // Inputs going into the final layer.
            Matrix final_inputs = Matrix.DotProduct(hidden_outputs, weightHiddenOutput);
            // Inputs leaving the final layer sigmoided.
            Matrix final_outputs = final_inputs.Sigmoid();

            return final_outputs;
        }

        /// <summary>
        /// Creates a two-dimensional matrix and fills it with doubles in a [-0.5,0.5] range.
        /// </summary>
        /// <param name="r"> Matrix rows </param>
        /// <param name="c"> Matrix columns </param>
        /// <returns> Matrix filled with doubles </returns>
        private Matrix RandomizeWeights(int r, int c)
        {
            Random rand = new Random();
            Matrix temp = new Matrix(r, c);
            for (int i = 0; i < temp.GetLength(0); i++)
                for (int j = 0; j < temp.GetLength(1); j++)
                    temp[i, j] = rand.NextDouble() - 0.5;
            return temp;
        }

        /// <summary>
        /// The primary training loop for the network. Updates weights based on the error
        /// of inputs and the target.
        /// </summary>
        /// <param name="inputs"> Inputs for the network. </param>
        /// <param name="target"> Target for the network. </param>
        public void train(Matrix inputs, Matrix target)
        {

            Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " is training...");
            // Inputs going into the hidden layer.
            Matrix hidden_inputs = Matrix.DotProduct(inputs, weightInputHidden);
            // Inputs leaving the hidden layer sigmoided.
            Matrix hidden_outputs = hidden_inputs.Sigmoid();

            // Inputs going into the final layer.
            Matrix final_inputs = Matrix.DotProduct(hidden_outputs, weightHiddenOutput);
            // Inputs leaving the final layer sigmoided.
            Matrix final_outputs = final_inputs.Sigmoid();

            // Errors in the output layer.
            Matrix output_errors = target - final_outputs;
            // Errors in the hidden layer.
            Matrix hidden_errors = Matrix.DotProduct(output_errors, weightHiddenOutput.Transpose());

            // Update the weights based on the previous calculations
            UpdateWeights(weightHiddenOutput, output_errors, final_outputs, hidden_outputs);
            UpdateWeights(weightInputHidden, hidden_errors, hidden_outputs, inputs);
        }

        /// <summary>
        /// Updates the weights using mathemagics.
        /// </summary>
        /// <param name="weights"> Weights to be updated. </param>
        /// <param name="errors"> Severity of errors in the network. </param>
        /// <param name="first_outputs"> First outputs. </param>
        /// <param name="second_outputs"> Second outputs. </param>
        private void UpdateWeights(Matrix weights, Matrix errors, Matrix first_outputs, Matrix second_outputs)
        {
            Matrix mathemagic = Matrix.DotProduct(second_outputs.Transpose(), errors * first_outputs * (1.0 - first_outputs));
            for (int i = 0; i < weights.GetLength(0); i++)
                for (int j = 0; j < weights.GetLength(1); j++)
                    weights[i, j] += mathemagic[i, j] * learningrate;
        }

        private void Trainloop(int start, int slice, System.Collections.Generic.List<String> data, NeuralNetwork network)
        {
            Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " is born!");
            int label;
            for (int i = start; i < start+slice-1; i++)
            {
                Console.WriteLine(Thread.CurrentThread.ManagedThreadId + " is advancing!");
                string line = data[i];
                string[] values = line.Split(',');
                Matrix inputs = new Matrix(1, 784);

                label = int.Parse(values[0]);
                for (int j = 1; j < values.Length - 1; j++)
                {
                    inputs[0, j] = (int.Parse(values[j]) / 255.0 * 0.99) + 0.01;
                }
                Matrix targets = new Matrix(1, 10);
                for (int k = 0; k < targets.GetLength(1); k++)
                {
                    targets[0, k] = 0.01;
                }
                
                targets[0, label] = 0.99;

                network.train(inputs, targets);
            }
        }

        /// <summary>
        /// Main for testing.
        /// </summary>
        /// <param name="args"> Not in use. </param>
        static void Main(string[] args)
        {
            NeuralNetwork test = new NeuralNetwork(INPUTNODES, HIDDENNODES, OUTPUTNODES, LEARNINGRATE);
            System.Collections.Generic.List<String> temp = new System.Collections.Generic.List<string>(60000);
            using (System.IO.StreamReader reader = new System.IO.StreamReader(@"mnist_train.csv"))
            {
                while (!reader.EndOfStream)
                {
                    temp.Add(reader.ReadLine());
                }
            }
            
            int processors = Environment.ProcessorCount;
            int slice = temp.Count / processors;
            for (int i = 0, j = 0; i < processors-1; i++, j+=slice) 
            {
                Thread newThread = new Thread(() => test.Trainloop(j, slice, temp, test));
                newThread.Start();
            }
            


           /* using (System.IO.StreamReader reader = new System.IO.StreamReader(@"mnist_train.csv"))
            {
                
                

                int label;
                while (!reader.EndOfStream)
                {
                    string line = reader.ReadLine();
                    string[] values = line.Split(',');
                    Matrix inputs = new Matrix(1, 784);

                    label = int.Parse(values[0]);
                    for (int i = 1; i < values.Length - 1; i++)
                    {
                        inputs[0, i] = (int.Parse(values[i]) / 255.0 * 0.99) + 0.01;
                    }
                    Matrix targets = new Matrix(1, 10);
                    for (int i = 0; i < targets.GetLength(1); i++)
                    {
                        targets[0, i] = 0.01;
                    }
                    targets[0, label] = 0.99;

                    test.train(inputs, targets);
                }
            }

            using (System.IO.StreamReader reader = new System.IO.StreamReader(@"mnist_test.csv"))
            {
                int label;
                string line = reader.ReadLine();
                string[] values = line.Split(',');
                Matrix inputs = new Matrix(1, 784);

                label = int.Parse(values[0]);
                for (int i = 1; i < values.Length - 1; i++)
                {
                    inputs[0, i] = (int.Parse(values[i]) / 255.0 * 0.99) + 0.01;
                }

                Matrix trueresults = test.Query(inputs);
            }*/
        }
    }
}
