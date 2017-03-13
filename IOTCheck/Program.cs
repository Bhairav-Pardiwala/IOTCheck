using CNTK;
using CNTKTest;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace IOTCheck
{
    class Program
    {
        static void Main(string[] args)
        {
            if(args.Length==2)
            {
               List<DenseVal> output= EvaluationSingleImage(DeviceDescriptor.DefaultDevice(), args[1]);

                DenseVal actualVal = null;
                double maxperc=0;
                foreach(var item in output)
                {
                    if(Convert.ToDouble(item.percent)>maxperc)
                    {
                        maxperc = (Convert.ToDouble(item.percent));
                        actualVal = item;
                    }
                }
                if(actualVal!=null)
                {
                    OutputFile(args[0], actualVal.val);
                }
                else
                {
                    OutputFile(args[0], "invalid");
                }

            }
            else
            {
                OutputFile(args[0], "invalid params");
            }

        }
        public static void OutputFile(String filename,String message)
        {
            using (StreamWriter outputFile = new StreamWriter(filename))
            {
                    outputFile.WriteLine(message);
            }
        }
        public static List<DenseVal> EvaluationSingleImage(DeviceDescriptor device,String fileName)
        {
            const string outputName = "z";
            var inputDataMap = new Dictionary<Variable, Value>();

            // Load the model.
            Function modelFunc = Function.LoadModel(@"C:\local\cntk\Tutorials\ImageHandsOn\Models\cifar10.cmf", device);

            // Get output variable based on name
            Variable outputVar = modelFunc.Outputs.Where(variable => string.Equals(variable.Name, outputName)).Single();

            // Get input variable. The model has only one single input.
            // The same way described above for output variable can be used here to get input variable by name.
            Variable inputVar = modelFunc.Arguments.First();
            var outputDataMap = new Dictionary<Variable, Value>();
            Value inputVal, outputVal;
            List<List<float>> outputBuffer;

            // Get shape data for the input variable
            NDShape inputShape = inputVar.Shape;
            uint imageWidth = inputShape[0];
            uint imageHeight = inputShape[1];
            uint imageChannels = inputShape[2];
            uint imageSize = inputShape.TotalSize;

            Console.WriteLine("Evaluate single image");

            // Image preprocessing to match input requirements of the model.
            Bitmap bmp = new Bitmap(Bitmap.FromFile(fileName));
            var resized = bmp.Resize((int)imageWidth, (int)imageHeight, true);
            List<float> resizedCHW = resized.ParallelExtractCHW();

            // Create input data map
            inputVal = Value.CreateBatch(inputVar.Shape, resizedCHW, device);
            inputDataMap.Add(inputVar, inputVal);

            // Create ouput data map. Using null as Value to indicate using system allocated memory.
            // Alternatively, create a Value object and add it to the data map.
            outputDataMap.Add(outputVar, null);

            // Start evaluation on the device
            modelFunc.Evaluate(inputDataMap, outputDataMap, device);

            // Get evaluate result as dense output
            outputBuffer = new List<List<float>>();
            outputVal = outputDataMap[outputVar];
            outputVal.CopyVariableValueTo(outputVar, outputBuffer);
            var s = PrintOutput(outputVar.Shape.TotalSize, outputBuffer);
            return s;

        }
        private static List<DenseVal> PrintOutput<T>(uint sampleSize, List<List<T>> outputBuffer)
        {
            Console.WriteLine("The number of sequences in the batch: " + outputBuffer.Count);
            int seqNo = 0;
            uint outputSampleSize = sampleSize;

            System.Collections.Generic.List<DenseVal> ListOfOutputs = new System.Collections.Generic.List<DenseVal>();


            foreach (var seq in outputBuffer)
            {
                Console.WriteLine(String.Format("Sequence {0} contains {1} samples.", seqNo++, seq.Count / outputSampleSize));
                uint i = 0;
                uint sampleNo = 0;

                uint cnt = 0;
                foreach (var element in seq)
                {

                    if (i++ % outputSampleSize == 0)
                    {
                        Console.Write(String.Format("    sample {0}: ", sampleNo));
                    }

                    String val = "";
                    switch (cnt)
                    {
                        case 0:
                         //   val = "airplane";
                            val = "closed";
                            break;
                        case 1:
                            //val = "automobile";
                            val = "open";
                            break;
                        //case 2:
                        //   // val = "bird";
                        //    break;
                        //case 3:
                        //    val = "cat";
                        //    break;
                        //case 4:
                        //    val = "deer";
                        //    break;
                        //case 5:
                        //    val = "dog";
                        //    break;
                        //case 6:
                        //    val = "frog";
                        //    break;
                        //case 7:
                        //    val = "horse";
                        //    break;
                        //case 8:
                        //    val = "ship";
                        //    break;

                        //case 9:
                        //    val = "truck";
                        //    break;

                    }

                    ListOfOutputs.Add(new DenseVal { percent = element.ToString(), val = val });
                    Console.Write(element);
                    if (i % outputSampleSize == 0)
                    {
                        Console.WriteLine(".");
                        sampleNo++;
                    }
                    else
                    {
                        Console.Write(",");
                    }
                    cnt += 1;
                }

            }
            return ListOfOutputs;



        }
    }
}
