using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public class InputStandardizingParameters : IAdditionalLayerParameters
    {
        public LayerType LayerType => LayerType.InputStandardizing;

        public InputStandardizingParameters()
        {
        }
    }
}
