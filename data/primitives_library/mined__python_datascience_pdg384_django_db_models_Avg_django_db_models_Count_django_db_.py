# MINED PRIMITIVE (Focus 1 co-occurrence pipeline)
# pid: python-datascience::pdg384::django.db.models.Avg+django.db.models.Count+django.db.models.Q
# name: django_primitive
# summary: Uses django.db.models.Avg, django.db.models.Count, django.db.models.Q across 3 repos
# anchor_symbols: ['django.db.models.Avg', 'django.db.models.Count', 'django.db.models.Q']
# observed in 3 repos: ['okfn-brasil__serenata-de-amor', 'polyaxon__haupt', 'practical-recommender-systems__moviegeek']...

# --- from practical-recommender-systems__moviegeek::recs/popularity_recommender.py::PopularityBasedRecs.predict_score ---
def predict_score(self, user_id, item_id):
        avg_rating = Rating.objects.filter(~Q(user_id=user_id) &
                                           Q(movie_id=item_id)).values('movie_id').aggregate(Avg('rating'))
        return avg_rating['rating__avg']

# --- from practical-recommender-systems__moviegeek::recs/popularity_recommender.py::PopularityBasedRecs.recommend_items ---
def recommend_items(self, user_id, num=6):
        pop_items = Rating.objects.filter(~Q(user_id=user_id)).values('movie_id').annotate(Count('user_id'),
                                                                                           Avg('rating'))
        sorted_items = sorted(pop_items, key=lambda item: -float(item['user_id__count']))[:num]
        return sorted_items

# --- from polyaxon__haupt::haupt/haupt/db/managers/stats.py::collect_entity_run_stats ---
def collect_entity_run_stats(**filters):
    data = (
        Models.Run.all.filter(**filters)
        .values("live_state")
        .annotate(
            run_count=Count("id"),
            sum_duration=Sum("duration"),
            sum_wait_time=Sum("wait_time"),
            sum_cpu=Sum("cpu"),
            sum_memory=Sum("memory"),
            sum_gpu=Sum("gpu"),
            sum_cost=Sum("cost"),
            sum_custom=Sum("custom"),
        )
    )
    rolling = collect_entity_run_rolling_stats(**filters)
    run_count = {item["live_state"]: item["run_count"] for item in data}
    tracking_time = {item["live_state"]: item["sum_duration"] for item in data}
    tracking_time["rolling"] = rolling.get("duration", {})
    wait_time = {item["live_state"]: item["sum_wait_time"] for item in data}
    wait_time["rolling"] = rolling.get("wait_time", {})
    cpu = {item["live_state"]: item["sum_cpu"] for item in data}
    cpu["rolling"] = rolling.get("cpu", {})
    memory = {item["live_state"]: item["sum_memory"] for item in data}
    memory["rolling"] = rolling.get("memory", {})
    gpu = {item["live_state"]: item["sum_gpu"] for item in data}
    gpu["rolling"] = rolling.get("gpu", {})
    cost = {item["live_state"]: item["sum_cost"] for item in data}
    cost["rolling"] = rolling.get("cost", {})
    custom = {item["live_state"]: item["sum_custom"] for item in data}
    custom["rolling"] = rolling.get("custom", {})
    resources = {
        "cpu": cpu,
        "memory": memory,
        "gpu": gpu,
        "cost": cost,
        "custom": custom,
    }
    return ProjectRunStats(run_count, tracking_time, wait_time, resources)

# --- from polyaxon__haupt::haupt/haupt/db/managers/stats.py::collect_entity_run_rolling_stats ---
def collect_entity_run_rolling_stats(**filters):
    last_time = now() - timedelta(days=30)
    duration_filter = Q(duration__gt=0)
    wait_time_filter = Q(duration__gt=0)
    cpu_filter = Q(cpu__gt=0)
    memory_filter = Q(memory__gt=0)
    gpu_filter = Q(gpu__gt=0)
    cost_filter = Q(cost__gt=0)
    custom_filter = Q(custom__gt=0)
    _data = Models.Run.all.filter(created_at__gte=last_time, **filters).aggregate(
        avg_duration=Avg("duration", filter=duration_filter),
        min_duration=Min("duration", filter=duration_filter),
        max_duration=Max("duration", filter=duration_filter),
        avg_wait_time=Avg("wait_time", filter=wait_time_filter),
        min_wait_time=Min("wait_time", filter=wait_time_filter),
        max_wait_time=Max("wait_time", filter=wait_time_filter),
        avg_cpu=Avg("cpu", filter=cpu_filter),
        min_cpu=Min("cpu", filter=cpu_filter),
        max_cpu=Max("cpu", filter=cpu_filter),
        avg_memory=Avg("memory", filter=memory_filter),
        min_memory=Min("memory", filter=memory_filter),
        max_memory=Max("memory", filter=memory_filter),
        avg_gpu=Avg("gpu", filter=gpu_filter),
        min_gpu=Min("gpu", filter=gpu_filter),
        max_gpu=Max("gpu", filter=gpu_filter),
        avg_cost=Avg("cost", filter=cost_filter),
        max_cost=Max("cost", filter=cost_filter),
        min_cost=Min("cost", filter=cost_filter),
        avg_custom=Avg("custom", filter=custom_filter),
        min_custom=Min("custom", filter=custom_filter),
        max_custom=Max("custom", filter=custom_filter),
    )
    data = {}
    for key in [
        "duration",
        "wait_time",
        "cpu",
        "memory",
        "gpu",
        "cost",
        "custom",
    ]:
        key_data = {
            "avg": _data.get(f"avg_{key}", 0),
            "min": _data.get(f"min_{key}", 0),
            "max": _data.get(f"max_{key}", 0),
        }
        data[key] = key_data

    return data

# --- from okfn-brasil__serenata-de-amor::jarbas/dashboard/admin/__init__.py::ReimbursementSummaryModelAdmin.get_cached_context ---
def get_cached_context(self, request, queryset):
        url = request.build_absolute_uri()
        hashed = md5(url.encode('utf-8')).hexdigest()
        key = f'cached_reimbursement_summary_context_{hashed}'
        context = cache.get(key)

        if context is not None:
            return context

        metrics = {
            'total_reimbursements': Count('id'),
            'total_value': Sum('total_net_value'),
        }
        queryset = (
            queryset
            .values('subquota_description')
            .annotate(**metrics)
            .order_by('-total_value')
        )

        chart_grouping = self.get_chart_grouping(request)
        if chart_grouping == 'year':
            chart_grouping_key = 'year'
            summary_over_time = (
                queryset
                .values('year')
                .annotate(total=Sum('total_net_value'))
                .order_by('year')
            )
        else:
            chart_grouping_key = 'chart_grouping'
            summary_over_time = (
                queryset
                .annotate(chart_grouping=Concat('year', 'month'))
                .values('chart_grouping')
                .annotate(total=Sum('total_net_value'))
                .order_by('year', 'month')
            )

        summary_over_time = tuple(summary_over_time)
        totals = tuple(row['total'] for row in summary_over_time)
        over_time_args = {
            'chart_grouping': chart_grouping,
            'chart_grouping_key': chart_grouping_key,
            'low': min(totals, default=0),
            'high': max(totals, default=0)
        }

        context = {
            'year': request.GET.get('year'),
            'month': request.GET.get('month'),
            'chart_grouping': chart_grouping,
            'summary': tuple(queryset),
            'summary_total': dict(queryset.aggregate(**metrics)),
            'summary_over_time': tuple(
                self.serialize_summary_over_time(row, **over_time_args)
                for row in summary_over_time
            )
        }

        cache.set(key, context, 60 * 60 * 6)
        return context
