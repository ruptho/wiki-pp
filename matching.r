library("dplyr")
library('MatchIt')
library("marginaleffects")
library('tidyr')
library('ggplot2')
library('fixest')

options(repr.plot.width = 15, repr.plot.height = 5)

propagate_labels_to_control <- function(paired_matches) {
    paired_matches <- paired_matches %>%
        group_by(subclass) %>%
        mutate(duration_days = max(duration_days[treated == TRUE], na.rm = TRUE)) %>%
        ungroup()
    
    paired_matches <- paired_matches %>%
      mutate(duration_label = case_when(
        duration_days %in% 1:6 ~ "Protected for 01-06 Days",
        duration_days == 7 ~ "Protected for 07 Days",
        duration_days %in% 8:92 ~ "Protected for 08-92 Days",
        #duration_days %in% 22:35 ~ "22-35 Days",
        #duration_days %in% 36:92 ~ "36-92 Days",
        TRUE ~ "Other"  # If none of the above conditions match
      ))
    
    paired_matches<- paired_matches %>%
        group_by(subclass) %>%
        mutate(Q_score_quantile = max(Q_score_quantile[treated == TRUE], na.rm = TRUE)) %>%
        ungroup()
    
    return(paired_matches)
}

merge_pairs_for_twfe <- function(paired_matches, df_did, time_var = "request_diff_week", 
                                 target_vars = c("revisions", "damaging_count", "goodfaith_count", 
                                                 "damaging", "goodfaith"), log_transform = FALSE) {

    paired_matches <- propagate_labels_to_control(paired_matches)
    
    df_matched_data_w <- merge_pairs_with_did(matches, df_week_data)
    df_matched_data_w <- df_matched_data_w[((df_matched_data_w[time_var] == 0) | (df_matched_data_w[time_var] == -1))]
    weekly_revs_around <- df_week_data %>%
          filter(between(request_diff_week, -4, 3)) %>%
          group_by(treated_id, request_diff_week) %>%
          summarize(revisions = sum(revisions)) %>%
          group_by(treated_id) %>%
          filter(all(revisions >= 1))
    
    df_matched_data_w <- merge(paired_matches, 
                               df_did[, c(c('treated_id', time_var), target_vars)],
                               by = "treated_id", all.x = TRUE, all.y = FALSE)
    df_matched_data_w$treated = df_matched_data_w$type == 'rfpp-protected'
    df_matched_data_w$post = df_matched_data_w[time_var] >= 0
    return(df_matched_data_w)
}

merge_pairs_with_did <- function(paired_matches, df_did, time_var = "request_diff_week", 
                                 target_vars = c("articlequality_last","articlequality_max", "articlequality_last_log", "page_size_max_log", "articlequality_last_norm", "page_size_last_log",
                                                "page_size_last_log_norm"), log_transform = FALSE) {
    # Calculate duration_days by subclass
    paired_matches <- propagate_labels_to_control(paired_matches)
    
    df_matched_data_w <- merge(paired_matches, 
                         df_did[, c(c('treated_id', time_var), target_vars)],
                         by = "treated_id", all.x = TRUE, all.y = FALSE)
    df_matched_data_w$treated = df_matched_data_w$type == 'rfpp-protected'
    df_matched_data_w$treated_int = as.integer(df_matched_data_w$treated)



    return(df_matched_data_w)
}

bootstrap_fixest <- function() {
}

analyze_matched_data <- function(df_matched_data_w, time_var = "request_diff_week", target_var = "articlequality_last", log_transform = FALSE, durations=c('07 Days'), topic=NULL, quality=NULL, show_plots=TRUE, min_val_week = -4, max_val_week = 13) {
  library(patchwork)
  cat('Retrieve protection with duration (', durations, ') and topic (', topic, ') and quality (', quality, ')\n')
  # Print row count
  #cat(length(df_matched_data_w[df_matched_data_w$treated, target_var]), 
   #    length(df_matched_data_w[!df_matched_data_w$treated, target_var]), '\n')

  # Drop rows with NA in the target_var
  df_matched_data_w <- df_matched_data_w %>% drop_na(target_var)
  
  # Print row count again
  #cat(length(df_matched_data_w[df_matched_data_w$treated, target_var]), 
  #    length(df_matched_data_w[!df_matched_data_w$treated, target_var]), '\n')
#
  # Create a variable for the log-transformed target variable
  if(log_transform) {
      log_target_var <- paste(target_var, "_log", sep = "")
      df_matched_data_w[log_target_var] <- log(df_matched_data_w[, target_var])
      target_var = log_target_var
  }
    
  # Log transformation
  max_val <- max_val_week
  min_val <- min_val_week
  if(time_var == 'request_diff_day') {
      min_val <- min_val * 7
      max_val <- max_val * 7
  }

  line_test <- df_matched_data_w
  line_test <- filter(line_test, line_test[time_var] >= min_val & line_test[time_var] < max_val)
 # cat(length(line_test[line_test$treated, target_var]),
    #  length(line_test[!line_test$treated, target_var]), '\n')

  if(length(durations) > 0) {
      line_test <- filter(line_test, duration_label %in% durations)
  } 
  if(!is.null(topic)) {
      line_test <- filter(line_test, .data[[topic]])
  }
  if(!is.null(quality)) {
      line_test <- filter(line_test, Q_score_quantile == quality)
  }
  # Replace matrix-based subsetting with one-dimensional logical vectors
  #cat(length(line_test[line_test$treated, target_var]),
     # length(line_test[!line_test$treated, target_var]), '\n')
    
  # Aggregate data
  grouped_data <- aggregate(formula(paste(target_var, " ~ ", time_var, " + type")), data = line_test, mean)
  
  # Create a line plot
  plot1 <- ggplot(grouped_data,  
                  aes(x = !!sym(time_var), y = !!sym(target_var), color = type, group = type)) +
                  geom_line() +
                  labs(x = "Time Variable", y = paste("Mean", target_var)) +
                  ggtitle(paste("Mean", target_var, "by", time_var, "and Type")) +
                  theme_minimal()
  
  #return(line_test)

  # Summarize data
  result_df <- grouped_data %>%
    group_by_at(time_var) %>%
    summarize_at(.vars=target_var, list(diff = ~ .[type == "rfpp-protected"] - .[type == "rfpp-declined"]))
  # Create a second line plot
  plot2 <- ggplot(result_df, aes_string(x = time_var, y = 'diff')) +
    geom_line() +
    labs(x = "Time Variable", y = paste("Difference in", target_var)) +
    ggtitle(paste("Difference in", target_var))

  if(show_plots == TRUE) {
    print(plot1 + plot2)
  }

  return(line_test)
}

plot_ggiplot <- function(data, x_min, x_max, time_var='request_diff_week', title_var='articlequality_last', title=NULL) {
  library(ggiplot)
    
  if(time_var == 'request_diff_week') {
      x_label <- 'Difference (in Weeks) from Request for Page Protection'
  }
  else {
      x_label <- 'Difference (in Days) from Request for Page Protection'
  }

  if(is.null(title)) {
      title <- paste('Effect of Page Protection on', title_var)
  }
  # col=c('#1E88E5', '#FFC107', '#004D40'),
  return(ggiplot(data, col=c('#1E88E5', '#FFC107', '#004D40'),#aggr.eff='post',
          multi_style='dodge', ref.line = -.5, xlab=x_label,
          main=title) +
  scale_x_continuous(breaks = seq(x_min, x_max, by = 1), 
                     limits=c(x_min-.5, x_max+.5),
                     minor_breaks=seq(from = x_min, to = x_max+1, by = 1) - .5,
                     expand = c(.01, .01)) + 
  theme(panel.grid.minor.x = element_line(colour="black", size=0.3),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.y = element_blank()) + theme(plot.margin = margin(0, 0, 0, 0),
                                                     text = element_text(size=14)))
   #  scale_colour_brewer(palette = 'GnBu', aesthetics = c('colour', 'fill'))
}

create_dataset_by_duration <- function(data, durations_list, time_var = "request_diff_week", target_var = "articlequality_last", log_transform = FALSE, show_plots=TRUE) {
    return(lapply(seq_along(durations_list), function(i) {
        return(analyze_matched_data(data, time_var = time_var, target_var =target_var, 
                              log_transform = log_transform, durations = durations_list[[i]], 
                                    show_plots=show_plots))
    }))
}


create_dataset_by_duration_labels <- function(data, time_var = "request_diff_week", target_var = "articlequality_last", log_transform = FALSE, show_plots=TRUE) {
    duration_labels <- as.list(unique(data$duration_label))
    frames <- lapply(seq_along(duration_labels), function(i) {
        d_l <- duration_labels[[i]]
        return(analyze_matched_data(data, time_var = time_var, target_var =target_var, 
                              log_transform = log_transform, durations = d_l, 
                                    show_plots=show_plots))
    })
    return(setNames(frames, duration_labels))
}

create_dataset_by_duration_and_topics <- function(
    data, duration, topics, time_var = "request_diff_week", 
    target_var = "articlequality_last", quality=NULL, log_transform = FALSE, max_val_week=13, show_plots=TRUE) {
    
    frames <- lapply(seq_along(topics), function(i) {
        return(analyze_matched_data(
            data, time_var = time_var, target_var =target_var, 
            log_transform = log_transform, durations = duration, topic=topics[[i]],
            quality=quality, max_val_week=max_val_week, show_plots=show_plots))
    })
    return(setNames(frames, topics))
}

create_dataset_by_duration_and_quality <- function(
    data, duration, q_cols, time_var = "request_diff_week", 
    target_var = "articlequality_last", topic=NULL, log_transform = FALSE,  max_val_week=13, show_plots=TRUE) {
    frames <- lapply(seq_along(q_cols), function(i) {
        return(analyze_matched_data(
            data, time_var = time_var, target_var =target_var, 
            log_transform = log_transform, durations = duration, topic=topic, quality=q_cols[[i]],
            max_val_week=max_val_week, show_plots=show_plots))
    })
    return(setNames(frames, q_cols))
}


fit_did <-  function(data, target_var, temp_var) {
    # we follow the tutorial by: 
    # https://lrberge.github.io/fixest/articles/fixest_walkthrough.html#simple-difference-in-differences-twfe
    duration_labels <- names(data)
    formula_vars <- "revisions_productive_1H_log  + revisions_productive_24H_log  + revisions_productive_168H_log + identity_reverts_24H_log + identity_reverts_1H_log  + identity_reverts_168H_log + Max_revision_text_bytes_log + page_age_log +"
    did_list <- lapply(seq_along(duration_labels), function(i) {
        f <- formula(paste(target_var, "~" , "  i(", temp_var, ", treated, -1)",
                           "| ", temp_var, " + treated_id"))
        return(feols(f, vcov='hc1', data=data[[duration_labels[i]]]))
        # ~subclass+treated_id
    })
    did_dict <- setNames(did_list, duration_labels)
}


fit_did_split <-  function(data, target_var, temp_var) {
    f <- formula(paste(target_var, " ~ i(", temp_var, ", treated, -1) | ", temp_var, " + treated_id "))
    return(feols(f, vcov=~subclass+treated_id, split=~duration_days, data=data))
}

aggregate_att <- function(did_list, names, day_regex="request_diff_week::[^-]") {
    agg_results <- do.call(rbind, lapply(seq_along(names), function(i) {
        return(aggregate(did_list[[i]],  agg= c("ATT" = day_regex)))
    }))
    rownames(agg_results) <- names
    return(agg_results)
}


extract_treated_coefs <- function(data) {
    coef_regex <- 'treatedTRUE:afterTRUE'
    ext_coefs <- coef(data, coef_regex)
    ext_cis <- confint(data, coef_regex, se='hc1')
    ext_cis$coef <- ext_coefs

    return(ext_cis)
}

extract_treated_coefs_for_list <- function(data) {
    results_list <- lapply(names(data), function(name) {
      data <- data[[name]]
      result <- extract_treated_coefs(data)  # Apply your function to the data frame
      result$Feature <- name  # Add a new column with the data frame name
      return(result)
    })  # Use simplify = FALSE to preserve names


    # Combine the results into a single data frame
    combined_results <- do.call(rbind, results_list)
    rownames(combined_results) <- NULL
    return(combined_results)
}

extract_treated_coefs_for_weeks <- function(data_list) {
  results_list <- lapply(names(data_list), function(model_name) {
    result <- extract_treated_coefs_for_list(data_list[[model_name]])
    result$Week <- model_name
    return(result)
  })
  
  # Combine the results into a single data frame
  combined_results <- do.call(rbind, results_list)
  
  # Reset row names to NULL
  rownames(combined_results) <- NULL
  
  return(combined_results)
}

extract_treated_coefs_for_quality <- function(data_list) {
  results_list <- lapply(names(data_list), function(model_name) {
    result <- extract_treated_coefs_for_weeks(data_list[[model_name]])
    result$Quality <- model_name
    return(result)
  })
  
  # Combine the results into a single data frame
  combined_results <- do.call(rbind, results_list)
  
  # Reset row names to NULL
  rownames(combined_results) <- NULL
  
  return(combined_results)
}



extract_start_end_coefs <- function(data, temp_var = 'request_diff_week', min_val=0, max_val=12) {
    coef_regex <- paste0(temp_var, '::(', min_val, '|(', max_val, ')):treated')
    ext_coefs <- coef(data, coef_regex)
    ext_cis <- confint(data, names(ext_coefs), se='hc1')
    ext_cis$coef <- ext_coefs
    ext_cis[temp_var] <- c(min_val, max_val)
    return(ext_cis)
}

extract_start_end_coefs_for_list <- function(data, temp_var = 'request_diff_week', min_val=0, max_val=12) {
    results_list <- lapply(names(data), function(name) {
      data <- data[[name]]
      result <- extract_start_end_coefs(data)  # Apply your function to the data frame
      result$category <- name  # Add a new column with the data frame name
      return(result)
    })  # Use simplify = FALSE to preserve names


    # Combine the results into a single data frame
    combined_results <- do.call(rbind, results_list)
    rownames(combined_results) <- NULL
    return(combined_results)
}


extract_start_end_coefs_for_models <- function(data_list, temp_var = 'request_diff_week', min_val=0, max_val=12) {
  results_list <- lapply(names(data_list), function(model_name) {
    result <- extract_start_end_coefs_for_list(data_list[[model_name]], temp_var = "request_diff_week")
    result$model <- model_name
    return(result)
  })
  
  # Combine the results into a single data frame
  combined_results <- do.call(rbind, results_list)
  
  # Reset row names to NULL
  rownames(combined_results) <- NULL
  
  return(combined_results)
}




export_table <- function(did_list, filename) {
    table_res <- etable(did_weekly_high, tex=TRUE, 
                 dict=c(treated='protected',
                        request_diff_week='week',
                        articlequality_last_log='Articlequality'))
    fileConn<-file(paste0('figures/tables/', filename, '.tex'))
    writeLines(table_res, fileConn)
    close(fileConn)
}