# *=========================================================================
# *
# *  Copyright Amsterdam UMC and contributors
# *
# *  Licensed under the GNU GENERAL PUBLIC LICENSE Version 3;
# *  you may not use this file except in compliance with the License.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *=========================================================================*/

import numpy as np
import pandas as pd

import snowphlake.mixture_model as mixture_model
import snowphlake.mallows_model as mallows_model
import snowphlake.utils as utils

## Todo: 
# 1. Basic model with fixed control Gaussians
# 2. Basic model with free control Gaussian
# 3. Basic model with visualizations for ordering, staging, event-centers, staging with kde
# 4. Snowphlake
# Coinit model with fixed control Gaussian (do this later)
# Coinit model with free control Gaussian (do this later)

class timeline():

    def __init__(self,confounding_factors=None, 
                    labels=None, bootstrap=False, flag_healthycontrols=True):
        self.confounding_factors = confounding_factors
        self.labels = labels 
        self.flag_healthycontrols = flag_healthycontrols
        self.bootstrap = bootstrap

    def estimate(self, data, diagnosis):

        if self.labels is not None:
            diagnosis=utils.set_diagnosis(diagnosis,self.labels)
        if self.confounding_factors is not None:
            ccf=utils.correct_confounding_factors()
            ccf.fit(data,diagnosis,self.confounding_factors)
            self.confounding_factors_model = ccf 
            data_corrected = self.confounding_factors_model.predict(data)
        else:
            self.confounding_factors_model = None 
            data_corrected = data
        self.n_biomarkers = data.shape[1]

        gmm = mixture_model.gaussian()
        gmm.fit(data_corrected,diagnosis,self.flag_healthycontrols)
        p_yes=gmm.predict_posterior(data_corrected)

        self.mixture_model = gmm 
    
        T = mallows_model()
        T.fit(p_yes)

        self.mallows_model = T 
        return

    def predict(self, data):

        utils.checkifestimated(self)
        data_corrected = self.confounding_factors_model.predict(data)
        p_yes=self.gmm.predict_posterior(data_corrected)
        stages = self.mallows_model.predict(p_yes)

        return stages,p_yes