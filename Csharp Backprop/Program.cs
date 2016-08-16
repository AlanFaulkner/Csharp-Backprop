using System;
using System.Collections.Generic;

namespace Csharp_Backprop
{

   
    internal class Program
    {
        private static void Main(string[] args)
        {
            Console.Write("Implementation of backpropagation algorithm V1.0.\n");
            ANN Network = new ANN();
            List<int> Net = new List<int> { 2, 1 };

            // Network.Create_Network(ref Net,1);
            //Network.Print_Weights();
            //Network.Save_Network("test.txt");
            List<double> Data = new List<double> { 0, 0 };
            Network.Load_Network("ForC.net");
                       
            List<List<double>> Test = new List<List<double>> {
                new List<double> {0,0 },
                new List<double> {0,1 },
                new List<double> {1,0 },
                new List<double> {1,1 },
            };

            List<List<double>> Out = new List<List<double>> {
                new List<double> {0 },
                new List<double> {1 },
                new List<double> {1 },
                new List<double> {1 },
            };
            Network.Back_prop_Stochastic(Test,Out);
            Network.Network_Validation(Test, Out);
           

            }
    }
}