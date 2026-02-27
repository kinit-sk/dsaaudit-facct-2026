import pandas as pd
from typing import List, Dict, Any
from statsmodels.stats.proportion import proportions_ztest


def calculate_age(date_of_birth: pd.Series) -> int:
	"""
	Calculate age from date of birth.

	Args:
		date_of_birth: Series containing date of birth

	Returns:
		Age as integer
	"""
	today = pd.Timestamp("today")
	dob = pd.to_datetime(date_of_birth)

	# Calculate age considering full date (year, month, day)
	age = today.year - dob.dt.year
	# Subtract 1 if birthday hasn't occurred yet this year
	age = age - ((today.month < dob.dt.month) | ((today.month == dob.dt.month) & (today.day < dob.dt.day)))

	return int(age.mean())


def calculate_age_group(age: int) -> str:
	"""
	Determine age group from age.

	Args:
		age: Age as integer

	Returns:
		'Minor' if under 18, 'Adult' otherwise
	"""
	return "Minor" if age < 18 else "Adult"


def add_percentage_columns(
	df: pd.DataFrame,
	count_columns: List[str],
	total_per_row: pd.Series,
	result_data: Dict[str, Any],
	result_columns: List[str],
) -> None:
	"""
	Add interleaved count and percentage columns to result data.

	Args:
		df: Source DataFrame with count data
		count_columns: List of column names to process
		total_per_row: Series with totals for percentage calculation
		result_data: Dictionary to append result data to (modified in place)
		result_columns: List to append column names to (modified in place)
	"""
	for col in count_columns:
		# Add count column
		result_columns.append(col)
		result_data[col] = df[col]

		# Add percentage column
		pct_col_name = f"{col}_pct"
		result_columns.append(pct_col_name)
		result_data[pct_col_name] = (df[col] / total_per_row * 100).round(2)
		result_data[pct_col_name] = result_data[pct_col_name].fillna(0)


def create_agent_metadata(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Create a mapping of agent metadata (Age, Age_Group, Interest, Gender).

	Args:
		df: DataFrame containing agent data

	Returns:
		DataFrame with agent as index and metadata columns
	"""
	agents = df["User"].unique()
	metadata = []

	for agent in agents:
		agent_df = df[df["User"] == agent]
		age = calculate_age(agent_df["date_of_birth"])
		metadata.append(
			{
				"agent": agent,
				"Age": age,
				"Age_Group": calculate_age_group(age),
				"Interest": agent_df["topic"].mode()[0],
				"Gender": agent_df["gender"].mode()[0],
			}
		)

	return pd.DataFrame(metadata).set_index("agent")


def produce_general_table(df: pd.DataFrame, agent_metadata: pd.DataFrame = None) -> pd.DataFrame:
	"""
	Produces a summary table for a given agent.

	Args:
		df: DataFrame containing the data
		agent_metadata: Optional pre-computed agent metadata DataFrame
	Returns:
		Summary DataFrame
	"""
	topics = df["topic"].unique().tolist() + ["other"]
	# Remove "politics" from topics as it will be counted as "other"
	topics = [t for t in topics if t != "politics"]
	agents = df["User"].unique()
	ad_types = [ad for ad in df["ad_type"].unique().tolist() if pd.notnull(ad)]

	# Build data dynamically using a dictionary
	data_dict = {}

	assert len(df["date_of_birth"].unique()) == len(
		df["User"].unique()
	), "Each agent must have a unique date_of_birth"
	assert len(df["gender"].unique()) == 2, "There should be exactly two unique genders in the data"
	assert len(df["topic"].unique()) >= 3, "There should be at least 3 unique topics in the data"

	# Create metadata if not provided
	if agent_metadata is None:
		agent_metadata = create_agent_metadata(df)

	# Get valid interest topics (beauty, fitness, gaming)
	valid_interest_topics = agent_metadata["Interest"].unique().tolist()

	for agent in agents:
		agent_df = df[df["User"] == agent]

		# Initialize agent column
		data_dict[agent] = {}

		# Add basic statistics from metadata
		data_dict[agent]["Age"] = agent_metadata.loc[agent, "Age"]
		data_dict[agent]["Age_Group"] = agent_metadata.loc[agent, "Age_Group"]
		data_dict[agent]["Gender"] = agent_metadata.loc[agent, "Gender"]
		data_dict[agent]["Interest"] = agent_metadata.loc[agent, "Interest"]
		valid_records = agent_df[agent_df["is_ad"].notnull()]
		data_dict[agent]["Total Records"] = valid_records.shape[0]
		data_dict[agent]["Non Ad Records"] = agent_df[agent_df["is_ad"] == False].shape[0]
		data_dict[agent]["Ads Detected"] = agent_df[agent_df["is_ad"] == True].shape[0]
		avg_video_length = valid_records["video_time_duration"].dropna().mean()
		data_dict[agent]["Average Video Length"] = round(avg_video_length, 2) if pd.notna(avg_video_length) else 0

		# Add ad type statistics
		for ad_type in ad_types:
			count = agent_df[(agent_df["is_ad"] == True) & (agent_df["ad_type"] == ad_type)].shape[0]
			data_dict[agent][f"{ad_type.capitalize()} Ads"] = count

			# Add topic-specific counts - iterate over ALL topics to avoid NaNs
			for ad_topic in topics:
				# Count all non-valid-interest topics as "other"
				if ad_topic == "other":
					# Count ads where ad_topic is not in valid interests OR is explicitly "other"
					topic_match_count = agent_df[
						(agent_df["is_ad"] == True)
						& (agent_df["ad_type"] == ad_type)
						& (~agent_df["ad_topic"].isin(valid_interest_topics))
					].shape[0]
				else:
					topic_match_count = agent_df[
						(agent_df["is_ad"] == True)
						& (agent_df["ad_type"] == ad_type)
						& (agent_df["ad_topic"] == ad_topic)
					].shape[0]
				data_dict[agent][f"{ad_type.capitalize()} Ads Topic: {ad_topic}"] = topic_match_count

	# Convert dictionary to DataFrame
	output_df = pd.DataFrame(data_dict)

	return output_df


def produce_topic_comparison_table(
	df: pd.DataFrame, agent_metadata: pd.DataFrame = None, ad_type_filter: str = "all"
) -> pd.DataFrame:
	"""
	Produces a topic comparison table across agents.

	Args:
		df: DataFrame containing the data
		agent_metadata: Optional pre-computed agent metadata DataFrame
		ad_type_filter: Filter for ad type - "formal", "influencer", "other", or "all" (default: "all")
	Returns:
		Topic comparison DataFrame
	"""
	# Calculate age for each agent
	df_copy = df.copy()

	# Filter for ads only
	df_ads = df_copy[df_copy["is_ad"] == True].copy()

	# Filter by ad_type if specified
	if ad_type_filter != "all":
		df_ads = df_ads[df_ads["ad_type"] == ad_type_filter]

	# Create metadata if not provided
	if agent_metadata is None:
		agent_metadata = create_agent_metadata(df)

	# Get valid interest topics (beauty, fitness, gaming)
	valid_interest_topics = agent_metadata["Interest"].unique().tolist()

	# Replace any topic that's not in valid interests with "other"
	df_ads["ad_topic"] = df_ads["ad_topic"].apply(lambda x: x if x in valid_interest_topics else "other")

	# Create a pivot table: agents as rows, ad topics as columns
	topic_comparison = df_ads.groupby(["User", "ad_topic"]).size().unstack(fill_value=0)

	# Add Interest and Age_Group columns from metadata
	topic_comparison.insert(0, "Age_Group", agent_metadata.loc[topic_comparison.index, "Age_Group"])
	topic_comparison.insert(1, "Interest", agent_metadata.loc[topic_comparison.index, "Interest"])

	# Add percentage columns for each ad topic
	ad_topics = [col for col in topic_comparison.columns if col not in ["Age_Group", "Interest"]]
	ad_topics = [topic for topic in ad_topics if topic in valid_interest_topics or topic == "other"]

	# Calculate total ads per agent (only for filtered topics)
	total_ads_per_agent = topic_comparison[ad_topics].sum(axis=1)

	# Create new dataframe with interleaved count and percentage columns
	result_columns = ["Age_Group", "Interest"]
	result_data = {
		"Age_Group": topic_comparison["Age_Group"],
		"Interest": topic_comparison["Interest"],
	}

	add_percentage_columns(topic_comparison, ad_topics, total_ads_per_agent, result_data, result_columns)

	result_df = pd.DataFrame(result_data, columns=result_columns)

	# Format percentage columns to exactly 2 decimal places
	pct_columns = [col for col in result_df.columns if col.endswith("_pct")]
	for col in pct_columns:
		result_df[col] = result_df[col].round(2)

	# Sort by Interest and Age_Group
	result_df = result_df.sort_values(by=["Interest", "Age_Group"])

	return result_df


def produce_personalization_comparison_table(
	df: pd.DataFrame, agent_metadata: pd.DataFrame = None, ad_type_filter="all"
) -> pd.DataFrame:
	"""
	Produces a personalization comparison table showing personalized vs non-personalized ads.

	Personalized: Ads of topic X seen by agents whose interest is X (in that demographic)
	Non-personalized: Ads of topic X seen by agents whose interest is NOT X (in that demographic)

	Args:
		df: DataFrame containing the data
		agent_metadata: Optional pre-computed agent metadata DataFrame
		ad_type_filter: Filter for ad type - "formal", "influencer", "other", "all" (default: "all"), or a list of ad types
	Returns:
		DataFrame with hierarchical structure comparing personalized vs non-personalized ads
	"""
	# Create metadata if not provided
	if agent_metadata is None:
		agent_metadata = create_agent_metadata(df)

	# Get valid interest topics
	valid_interest_topics = agent_metadata["Interest"].unique().tolist()

	# Filter for ads only
	df_ads = df[df["is_ad"] == True].copy()

	# Filter by ad_type if specified
	if ad_type_filter != "all":
		if isinstance(ad_type_filter, list):
			df_ads = df_ads[df_ads["ad_type"].isin(ad_type_filter)]
		else:
			df_ads = df_ads[df_ads["ad_type"] == ad_type_filter]

	# Replace any topic that's not in valid interests with "other"
	df_ads["ad_topic"] = df_ads["ad_topic"].apply(lambda x: x if x in valid_interest_topics else "other")

	# Merge with agent metadata to get Interest and Age_Group
	df_ads = df_ads.merge(agent_metadata[["Interest", "Age_Group"]], left_on="User", right_index=True)

	# Build the result data
	results = []

	for interest in sorted(valid_interest_topics):
		for age_group in ["Minor", "Adult"]:
			# Get agents with this interest and age group (for personalized)
			agents_with_interest = agent_metadata[
				(agent_metadata["Interest"] == interest) & (agent_metadata["Age_Group"] == age_group)
			].index

			# Get agents with different interest but same age group (for non-personalized)
			agents_without_interest = agent_metadata[
				(agent_metadata["Interest"] != interest) & (agent_metadata["Age_Group"] == age_group)
			].index

			# Skip if no agents with this interest in this demographic
			if len(agents_with_interest) == 0:
				continue

			# Get total ads (filtered by ad_type) for agents WITH this interest
			total_records_personalized = df_ads[df_ads["User"].isin(agents_with_interest)].shape[0]

			# Get total ads (filtered by ad_type) for agents WITHOUT this interest (but same demographic)
			total_records_non_personalized = df_ads[df_ads["User"].isin(agents_without_interest)].shape[0]

			# Count personalized ads: ads of this topic seen by agents who like this topic
			personalized_count = df_ads[
				(df_ads["User"].isin(agents_with_interest)) & (df_ads["ad_topic"] == interest)
			].shape[0]

			# Count non-personalized ads: ads of this topic seen by agents who DON'T like this topic
			non_personalized_count = df_ads[
				(df_ads["User"].isin(agents_without_interest)) & (df_ads["ad_topic"] == interest)
			].shape[0]

			# Calculate two-proportion z-test
			# Test if proportions are significantly different
			z_stat = None
			p_value = None
			significance = ""
			if total_records_personalized > 0 and total_records_non_personalized > 0:
				try:
					counts = [personalized_count, non_personalized_count]
					nobs = [total_records_personalized, total_records_non_personalized]
					z_stat, p_value = proportions_ztest(counts, nobs)
					z_stat = round(z_stat, 3)
					p_value = round(p_value, 4)

					# Determine significance stars
					if p_value < 0.001:
						significance = "***"
					elif p_value < 0.01:
						significance = "**"
					elif p_value < 0.05:
						significance = "*"
				except:
					pass

			# Add non-personalized row
			results.append(
				{
					"Category": f"{interest.capitalize()}",
					"Type": "Does not match topic of user",
					"Demographic": age_group.lower(),
					"Ratio from total ads": f"{non_personalized_count}/{total_records_non_personalized}"
					if total_records_non_personalized > 0
					else "0/0",
					"Percentage": round(
						(non_personalized_count / total_records_non_personalized * 100)
						if total_records_non_personalized > 0
						else 0,
						2,
					),
					"Z-statistic": "",
					"P-value": "",
					"Significance": "",
				}
			)

			# Add personalized row
			results.append(
				{
					"Category": "",
					"Type": "Matches topic of user",
					"Demographic": age_group.lower(),
					"Ratio from total ads": f"{personalized_count}/{total_records_personalized}"
					if total_records_personalized > 0
					else "0/0",
					"Percentage": round(
						(personalized_count / total_records_personalized * 100)
						if total_records_personalized > 0
						else 0,
						2,
					),
					"Z-statistic": z_stat if z_stat is not None else "",
					"P-value": p_value if p_value is not None else "",
					"Significance": significance,
				}
			)

	result_df = pd.DataFrame(results)

	return result_df


if __name__ == "__main__":
	data = pd.read_csv("facct_data.csv")

	# Create agent metadata mapping once
	agent_metadata = create_agent_metadata(data)
	agent_metadata.to_csv("agent_metadata.csv")
	print("\nAgent Metadata:")
	print(agent_metadata)

	# Display basic information
	print(f"\nTotal combined records: {data.shape[0]}")
	print(data["User"].value_counts())
	print(data["ad_type"].value_counts())

	# Produce and print summary table
	summary_table = produce_general_table(data, agent_metadata)
	summary_table.to_csv("summary_table.csv")
	print(summary_table)

	# Produce and print topic comparison table
	for ad_type in ["formal", "influencer", "other"]:
		print(f"\nTopic Comparison Table - Ad Type: {ad_type}")
		topic_comparison_table = produce_topic_comparison_table(
			data, agent_metadata, ad_type_filter=ad_type
		)
		topic_comparison_table.to_csv(f"topic_comparison_table_{ad_type}.csv")
		print(topic_comparison_table)

	# Produce and print personalization comparison table
	# 1. Formal ads
	print("\nPersonalization Comparison Table - Ad Type: formal")
	personalization_table = produce_personalization_comparison_table(
		data, agent_metadata, ad_type_filter="formal"
	)
	personalization_table.to_csv("personalization_comparison_table_formal.csv")
	print(personalization_table)

	# 2. Influencer + Other ads combined
	print("\nPersonalization Comparison Table - Ad Type: influencer+other")
	personalization_table = produce_personalization_comparison_table(
		data, agent_metadata, ad_type_filter=["influencer", "other"]
	)
	personalization_table.to_csv("personalization_comparison_table_influencer_other.csv")
	print(personalization_table)
