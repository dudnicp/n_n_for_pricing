using NeuralNetwork.Common.Layers;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Layers.AdditionalLayerParameters
{
    public interface IAdditionalLayerParameters
    {
        LayerType LayerType { get; }
    }
}
