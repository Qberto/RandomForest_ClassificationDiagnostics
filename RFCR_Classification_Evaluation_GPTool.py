import pandas as pd
import arcpy
import math

def convert_featureclass_to_pddataframe(fc, fields_list=["*"], remove_index=False):
    """
    Converts a feature class to a pandas dataframe.
    :param fc: Input feature class
    :param fields_list: Optional parameter - defaults to wildcard ["*"]. Optionally include specific fields.
    :param remove_index: Optional parameter - defaults to False. Remove index from output dataframe.
    :return: Pandas dataframe of the feature class WITHOUT geometry and date attributes.
    """
    # Gather a list of all field names if the user did not specify field inputs
    if fields_list == ["*"]:
        # Generate a valid list of field names that can be passed to the arcpy.FeatureClassToNumPyArray function
        # (must filter out geometry and date fields!
        fields_list = [field_object.name for field_object in arcpy.ListFields(fc) if field_object.type not in ["Geometry", "Date"]]

    temp_array = arcpy.da.FeatureClassToNumPyArray(fc, fields_list)
    df = pd.DataFrame(data=temp_array)
    if remove_index:
        df.reset_index(inplace=True)
    return df

# Helper function to return result boolean val
def return_boolean_result(known_val, predicted_val):

    if predicted_val == known_val:
        return 1
    else:
        return 0 

# Helper function to determine confusion matrix values
def return_confusion_matrix_val(known_val, predicted_val, positive_val, negative_val):
    # Set true positive
    if known_val == positive_val and predicted_val == positive_val:
        return "True Positive" 
    
    # Set true negative
    if known_val == negative_val and predicted_val == negative_val:
        return "True Negative"
    
    # Set false positive
    if known_val == negative_val and predicted_val == positive_val:
        return "False Positive" 
    
    # Set false negative
    if known_val == positive_val and predicted_val == negative_val:
        return "False Negative"  

def calculate_accuracy(tp, tn, fp, fn, round_val=4):
	acc = round((tp + tn) / (tp + tn + fp + fn), round_val)
	return acc

def calculate_fscore(tp, fp, fn, round_val=4):
	# Calculate precision and recall
	precision = tp / (tp+fp)
	recall = tp / (tp+fn)
	# Handle division by zero
	if precision == 0 and recall == 0:
		f = "N/A"
	else:
		# Calculate f-score
		f = round(2 * ( (precision * recall) / (precision + recall)), round_val)
	return f, precision, recall

def calculate_mcc(tp, tn, fp, fn, round_val=4):
	if tp+fp == 0 or tp+fn == 0 or tn+fp == 0 or tn+fn == 0:
		mcc = "N/A"
	else:
		mcc = round(((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)), round_val)
	return mcc

# Calculates the accuracy of the classification results from a run of "Random Forest Classification and Regression"
def evaluate_classification(predicted_fc, known_val_field, predicted_val_field, trained_field="TRAINED_ID", positive_val=1, negative_val=0, verbose=False):

    # Bring necessary modules
    import pandas as pd
    import arcgis

    # Extract the predicted feature class into a pandas dataframe
    # df = arcgis.features.SpatialDataFrame().from_featureclass(predicted_fc)
    df = convert_featureclass_to_pddataframe(predicted_fc)

    # Find all the records that were not used to train (trained_field == 0)
    df = df[(df[trained_field] == 0)]

    # Get a total count
    total_test_records = df.shape[0]
    
    # Compute a new boolean field 'result' and determine if prediction was correct
    df['CORRECT'] = df.apply(lambda x: return_boolean_result(x[known_val_field], x[predicted_val_field]), axis=1)

    # Get total count of correct answers
    total_correct_records = df[(df['CORRECT']==1)].shape[0]
   
    # Calculate confusion matrix vals
    df['CONFUSION_MATRIX_VAL'] = df.apply(lambda x: return_confusion_matrix_val(x[known_val_field], 
                                                                                x[predicted_val_field], 
                                                                                positive_val, 
                                                                                negative_val), axis=1) 
    
    # Calculate totals from confusion matrix
    total_tp = df[(df['CONFUSION_MATRIX_VAL'] == "True Positive")].shape[0]
    total_tn = df[(df['CONFUSION_MATRIX_VAL'] == "True Negative")].shape[0]
    total_fp = df[(df['CONFUSION_MATRIX_VAL'] == "False Positive")].shape[0]
    total_fn = df[(df['CONFUSION_MATRIX_VAL'] == "False Negative")].shape[0]

    # Calculate the derivations from the confusion matrix
    # percent_correct = round(total_correct_records / total_test_records, 4)
    accuracy = calculate_accuracy(total_tp, total_tn, total_fp, total_fn)
    fscore, precision, recall = calculate_fscore(total_tp, total_fp, total_fn)
    mcc = calculate_mcc(total_tp, total_tn, total_fp, total_fn)
    
    total_known_positive = df[(df[known_val_field]==positive_val)].shape[0]
    total_known_negative = df[(df[known_val_field]==negative_val)].shape[0]
    
    tp_perc = round(total_tp / total_known_positive, 4) * 100
    tn_perc = round(total_tn / total_known_negative, 4) * 100
    fp_perc = round(total_fp / total_known_positive, 4) * 100
    fn_perc = round(total_fn / total_known_negative, 4) * 100
    
    tp_total_perc = round(total_tp / total_test_records, 4) * 100
    tn_total_perc = round(total_tn / total_test_records, 4) * 100
    fp_total_perc = round(total_fp / total_test_records, 4) * 100
    fn_total_perc = round(total_fn / total_test_records, 4) * 100
    
    print(">>>> Classification Diagnostics <<<<")
    
    if verbose:
	    print("\nObservations: ")
	    print("\tTotal Known Positives: {0}".format(total_known_positive))
	    print("\tTotal Known Negatives: {0}".format(total_known_negative))
	    
	    print("\nClassification Results: ")
	    print("\tTrue Positives: {0} ({1}%)".format(total_tp, tp_perc))
	    print("\tTrue Negatives: {0} ({1}%)".format(total_tn, tn_perc))
	    print("\tFalse Positives: {0} ({1}%)".format(total_fp, fp_perc))
	    print("\tFalse Negatives: {0} ({1}%)".format(total_fn, fn_perc))


    print("\n\tPrecision: {0}".format(round(precision, 4)))
    print("\tRecall: {0}".format(round(recall, 4)))

    print("\n\tAccuracy: {0}".format(accuracy))
    print("\tF-Score: {0}".format(str(fscore)))
    print("\tMCC: {0}".format(str(mcc)))
    
    return accuracy, fscore, mcc

def select_true_positives(predicted_fc_lyr_name, known_val_field, predicted_val_field, trained_field="TRAINED_ID", positive_val=1):

	arcpy.management.SelectLayerByAttribute(predicted_fc_lyr_name, 
	                                        "NEW_SELECTION", 
	                                        "{0} = 0 And {1} = {3} And {2} = {3}".format(trained_field, 
	                                                                                     known_val_field, 
	                                                                                     predicted_val_field,
	                                                                                     positive_val), 
	                                        None)

def select_true_negatives(predicted_fc_lyr_name, known_val_field, predicted_val_field, trained_field="TRAINED_ID", negative_val=0):

	arcpy.management.SelectLayerByAttribute(predicted_fc_lyr_name, 
	                                        "NEW_SELECTION", 
	                                        "{0} = 0 And {1} = {3} And {2} = {3}".format(trained_field, 
	                                                                                     known_val_field, 
	                                                                                     predicted_val_field,
	                                                                                     negative_val), 
	                                        None)

def select_false_positives(predicted_fc_lyr_name, known_val_field, predicted_val_field, trained_field="TRAINED_ID", positive_val=1, negative_val=0):

	arcpy.management.SelectLayerByAttribute(predicted_fc_lyr_name, 
	                                        "NEW_SELECTION", 
	                                        "{0} = 0 And {1} = {3} And {2} = {4}".format(trained_field, 
	                                                                                     known_val_field, 
	                                                                                     predicted_val_field,
	                                                                                     positive_val,
	                                                                                     negative_val), 
	                                        None)

def select_false_negatives(predicted_fc_lyr_name, known_val_field, predicted_val_field, trained_field="TRAINED_ID", positive_val=1, negative_val=0):

	arcpy.management.SelectLayerByAttribute(predicted_fc_lyr_name, 
	                                        "NEW_SELECTION", 
	                                        "{0} = 0 And {1} = {4} And {2} = {3}".format(trained_field, 
	                                                                                     known_val_field, 
	                                                                                     predicted_val_field,
	                                                                                     positive_val,
	                                                                                     negative_val), 
	                                        None)	   


# known_val_field = "ARSON"
# predicted_val_field = "PREDICTED"
# trained_field = "TRAINED_ID"

# TODOs
# - Utility to determine imbalance
# - Utility to generate balanced input
# - Utility to transfer diagnostics output as fields to output dataset (fields with FP, FN, TP, TN) in order to symbolize, etc.
# 
# 


if __name__ == "__main__":
    # Establish general variables that will be determined by the user as input parameters
    predicted_fc = arcpy.GetParameterAsText(0)
    def evaluate_classification(predicted_fc, known_val_field, predicted_val_field, trained_field="TRAINED_ID", positive_val=1, negative_val=0, verbose=False):