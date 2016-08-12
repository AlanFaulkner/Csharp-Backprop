using System;
using System.Collections.Generic;

namespace Csharp_Backprop
{
    internal class Neuron
    {
        public List<double> Weights = new List<double>() { };
        public double Output;

        // Constructor
        public Neuron(int Number_Of_Connections, int Seed)
        {
            Random RN = new Random(Seed);

            for (int i = 0; i < Number_Of_Connections; i++)
            {
                Weights.Add(RN.NextDouble() * 2.0 - 1.0);
            }

            // Add additional value to represent bias
            Weights.Add(RN.NextDouble());
        }
    }

    internal class ANN
    {
        public List<List<Neuron>> Network = new List<List<Neuron>>() { };
        private List<int> Network_Information = new List<int>() { };

        // Core Functions
        public void Create_Network(ref List<int> Network_Description, int Seed)
        {
            Network_Information = Network_Description;

            if (Network_Description.Count < 2)
            {
                Console.Write("The description of neural net you have entered is not valid!\n\nA valid description must contain at least two values:\n   The number of inputs into the network\n   The number of output neurons\n\n In both cases the minimum value allowed is 1!\n\n");
                return;
            }

            for (int i = 0; i < Network_Description.Count; i++)
            {
                if (Network_Description[i] == 0)
                {
                    Console.Write("The description of neural net you have entered is not valid!\n\n The minimum allowed number of neurons or inputs is 1\n\n");
                    return;
                }
            }
            Random Rnd = new Random(Seed);
            for (int i = 1; i < Network_Description.Count; i++)
            {
                List<Neuron> Layer = new List<Neuron>() { };
                for (int j = 0; j < Network_Description[i]; j++)
                {
                    if (i == 1)
                    {
                        Neuron neuron = new Neuron(Network_Description[0], Rnd.Next());
                        Layer.Add(neuron);
                    }
                    else
                    {
                        Neuron neuron = new Neuron(Network_Description[i - 1], Rnd.Next());
                        Layer.Add(neuron);
                    }
                }
                Network.Add(Layer);
            }
            Console.Write("Network created successfully!\n\n");
            return;
        }

        public void Save_Network(string filename)
        {
            using (System.IO.StreamWriter Out = new System.IO.StreamWriter("../" + filename, false))
            {
                for (int i = 0; i < Network.Count; i++)
                {
                    for (int j = 0; j < Network[i].Count; j++)
                    {
                        for (int k = 0; k < Network[i][j].Weights.Count; k++)
                        {
                            Out.Write(Network[i][j].Weights[k] + Environment.NewLine);
                        }
                    }
                }
                Out.Flush();
                Out.Close();
            }
        }

        public void Load_Network(string filename)
        {
            string line;
            System.IO.StreamReader fs = new System.IO.StreamReader(@"../" + filename);
            while ((line = fs.ReadLine()) != null && line != "Data")
            {
                // convert string to int
                int x;
                Int32.TryParse(line, out x);
                Network_Information.Add(x);
            }

            // Check loaded data validity
            if (Network_Information.Count < 2)
            {
                Console.Write("The description of neural net you have entered is not valid!\n\nA valid description must contain at least two values:\n   The number of inputs into the network\n   The number of output neurons\n\n In both cases the minimum value allowed is 1!\n\n");
                return;
            }

            for (int i = 0; i < Network_Information.Count; i++)
            {
                if (Network_Information[i] < 1)
                {
                    Console.Write("The description of neural net you have entered is not valid!\n\nA valid description must contain at least two values:\n   The number of inputs into the network\n   The number of output neurons\n\n In both cases the minimum value allowed is 1!\n\n");
                    return;
                }
            }

            // Build network based on input data
            for (int i = 1; i < Network_Information.Count; i++)
            {
                List<Neuron> Layer = new List<Neuron>() { };
                for (int j = 0; j < Network_Information[i]; j++)
                {
                    if (i == 1)
                    {
                        Neuron neuron = new Neuron(Network_Information[0], 0);
                        for (int z = 0; z < neuron.Weights.Count; z++)
                        {
                            double x;
                            double.TryParse((line = fs.ReadLine()), out x);
                            neuron.Weights[z] = x;
                        }
                        Layer.Add(neuron);
                    }
                    else
                    {
                        Neuron neuron = new Neuron(Network_Information[i - 1], 0);
                        for (int z = 0; z < neuron.Weights.Count; z++)
                        {
                            double x;
                            double.TryParse((line = fs.ReadLine()), out x);
                            neuron.Weights[z] = x;
                        }
                        Layer.Add(neuron);
                    }
                }
                Network.Add(Layer);
            }

            fs.Close();
        }

        // Beta functions
        public List<double> Calculate_Netowrk_Output(ref List<double> Input_Data)
        {
            // version 1.0 only calculates a single data set output
            List<double> Output = new List<double> { };
            Input_Data.Add(1); // default input for the bias

            for (int i = 0; i < Network.Count; i++)
            {
                for (int j = 0; j < Network[i].Count; j++)
                {
                    // iterate through input data and multiple by the weights for each neuron and store the sum
                    // apply activation function to sum
                    // set neuron output to result.

                    double sum = 0;
                    for (int k = 0; k < Input_Data.Count; k++)
                    {
                        sum += Network[i][j].Weights[k] * Input_Data[k];
                    }

                    sum = Activation_Function(ref sum, "Sigmoid", false); // apply desired activation function
                    Network[i][j].Output = sum; // set the output value for neuron
                }
                // clear current input data and refill it with the outputs from previous layer to generate inputs into next layer.
                Input_Data.Clear();
                for (int j = 0; j < Network[i].Count; j++)
                {
                    Input_Data.Add(Network[i][j].Output);
                }
                Input_Data.Add(1); // Bias input
            }
            Output = Input_Data; // get output of network

            return Output;
        }

        private double Activation_Function(ref double X, string Function, bool Differential)
        {
            switch (Function)
            {
                case ("Identity"):
                    // Limits -inf -> inf
                    if (Differential == true) { return 1; }
                    else return X;

                case ("Sigmoid"):
                    // Limits 0 -> 1
                    if (Differential == true) { return X * (1 - X); }
                    else return 1 / (1 + Math.Exp(-X));

                case ("TanH"):
                    // Limits -1 -> 1
                    if (Differential == true) { return (4 * Math.Cosh(X) * Math.Cosh(X)) / ((Math.Cosh(2 * X) + 1) * (Math.Cosh(2 * X) + 1)); }
                    else return Math.Tanh(X);

                case ("ArcTan"):
                    // Limits -pi/2 -> pi/2
                    if (Differential == true) { return 1 / (X * X + 1); }
                    else return Math.Atan(X);
                    ;
                case ("Softsign"):
                    // Limits -1 -> 1
                    if (Differential == true) { return 1 / ((1 - Math.Abs(X)) * (1 - Math.Abs(X))); }
                    else return X / (1 - Math.Abs(X));

                case ("Sinusoid"):
                    // Limits -1 -> 1
                    if (Differential == true) { return Math.Cos(X); }
                    else return Math.Sin(X);

                case ("Gaussian"):
                    // Limits 0 -> 1
                    if (Differential == true) { return -2 * X * Math.Exp(-1 * X * X); }
                    else return Math.Exp(-1 * X * X);

                default:
                    return 1;
            }
        }

        // Diagnostic functions
        public void Print_Weights()
        {
            for (int i = 0; i < Network.Count; i++)
            {
                for (int j = 0; j < Network[i].Count; j++)
                {
                    Network[i][j].Weights.ForEach(Console.WriteLine);
                    Console.Write("\n\n");
                }
            }
        }

        public void Print_Outputs()
        {
            for (int i = 0; i < Network.Count; i++)
            {
                if (i != Network.Count - 1) { Console.Write("\nHidden layer " + i + Environment.NewLine); }
                else { Console.Write("\nOutput Layer\n\n"); }
                for (int j = 0; j < Network[i].Count; j++)
                {
                    Console.Write(Network[i][j].Output + Environment.NewLine);
                }
            }
        }
    }

    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.Write("Implementation of backpropagation algorithm V1.0.\n");
            ANN Network = new ANN();
            List<int> Net = new List<int> { 2, 2, 1 };

            //Network.Create_Network(ref Net,1);
            //Network.Print_Weights();
            //Network.Save_Network("test.txt");
            Console.Write("load\n\n");
            Network.Load_Network("test.txt");
            Network.Print_Weights();
            List<double> Result = new List<double> { };
            List<double> Data = new List<double> { 1, 1 };
            Result = Network.Calculate_Netowrk_Output(ref Data);
            // Console.WriteLine(Network.Network[0][0].Output);
            // Result.ForEach(Console.WriteLine);
            Network.Print_Outputs();
        }
    }
}