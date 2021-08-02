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

import pandas as pd 
import numpy as np 

def set_diagnosis(diagnosis,labels):

    idx_cn = diagnosis == labels[0]
    idx_ad = diagnosis == labels[-1]

    diagnosis[idx_cn]=1
    diagnosis[idx_ad]=3
    diagnosis[~np.logical_or(idx_cn,idx_ad)]=2

    return diagnosis

class correct_confounding_factors():

    def fit(self,data,diagnosis,confounding_factors):

        return
    
    def predict(self,data):
        
        return