using NeuralNetwork.Common.GradientAdjustmentParameters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkCreator.Services
{
    public static class AviableGradientAdjustmentTypesService
    {
        private static List<GradientAdjustmentType> _types;
        public static List<GradientAdjustmentType> Types
        {
            get
            {
                if (_types == null)
                {
                    _types = new List<GradientAdjustmentType>();
                    foreach (GradientAdjustmentType type in (GradientAdjustmentType[])Enum.GetValues(typeof(GradientAdjustmentType)))
                    {
                        _types.Add(type);
                    }
                }

                return _types;
            }
        }


    }
}
