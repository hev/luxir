// Copyright 2020-2026 Yonik Seeley and Luxir contributors
// SPDX-License-Identifier: Apache-2.0

#include "luxir/query/BooleanWeight.h"
#include "luxir/query/NumericPredicateQuery.h"
#include "luxir/query/TermQuery.h"
#include "luxir/search/Collector.h"
#include "test/CollectionHelper.h"
#include "test/ConjunctionPlanControl.h"
#include "test/NumericVerification.h"
#include "test/SchemaBuilder.h"
#include "test/TopKAssert.h"

using namespace luxir;
using namespace luxir::test;

namespace {

using Use = Query::ExecutionUse;
using Consumption = Query::MembershipConsumption;
using Answer = Query::ScorerSupplier::BulkAnswer;
using FilterConsumption = Query::ScorerSupplier::WindowFilterConsumption;

class RangeFilterTest : public LuxirTest {
protected:
  static constexpr int32_t N = 32768;
  CollectionHelper helper;
  std::shared_ptr<IndexReader> reader;

  static bool possible(int32_t doc) {
    return (doc >= 4096 && doc < 12288) || (doc >= 16384 && doc < 24576);
  }

  void SetUp() override {
    SchemaBuilder schema;
    for (auto name : {"points_i", "proof_points_i"}) {
      auto& field = schema.field(name);
      field.type = api::FieldDef::FieldClass::INT;
      field.index = api::FieldDef::IndexMode::RANGE;
    }
    schema.set(helper.collection());
    std::vector<Doc> docs;
    for (int32_t doc = 0; doc < N; doc++) {
      int64_t value = 2 * ((doc * 1543) % N);
      std::string body = "pad";
      if (doc % 64 == 0) body += " a";
      if (doc % 2048 == 0) body += " rare";
      if (doc % 256 == 0) body += " threshold";
      if (doc % 3 == 0) body += " b";
      if (doc % 64 == 1 || doc % 128 == 0) body += " c";
      if (doc % 16 == 0) body += " dense_a";
      if (doc % 32 == 0) body += " dense_b";
      if (doc % 4 == 0) body += " excluded";
      docs.push_back(flatdoc("id", std::to_string(doc), "body_w", body,
          "column_i", value, "points_i", value, "cluster_i", doc,
          "proof_i", possible(doc) ? doc : N + doc,
          "proof_points_i", possible(doc) ? doc : N + doc,
          "proof2_i", doc >= 8192 && doc < 20480 ? doc : N + doc,
          "partial_i", doc < N / 4 ? N + doc : value,
          "striped_i", (doc / 512) % 2 == 0 ? 0 : N));
      if (doc % 5 != 0) docs.back().push_back({"multi_is", vec_i(value, value + 1000)});
      if (doc % 5 != 0) docs.back().push_back({"holes_i", (int64_t)(possible(doc) ? doc : N + doc)});
      if (doc >= 2048 && doc < N - 2048 && doc % 5 != 0) {
        docs.back().push_back({"edge_i", (int64_t)(doc >= 8192 && doc < 12288 ? N + doc : doc)});
      }
      docs.back().push_back({"cluster_is", vec_i(doc, doc + 1)});
    }
    ASSERT_TRUE(helper.indexAll(docs, UpdateMessage::COMMIT).success);
    reader = std::make_shared<IndexReader>(helper.getIndexWriter()->dir);
    ASSERT_EQ(1u, reader->segments().size());
  }

  std::vector<int32_t> membership(Query& query, Use use, DocSet* domain = nullptr,
                                  std::span<Query*> enclosing = {}, bool reposition = false,
                                  int32_t min = 0, int32_t max = N) {
    MemPool pool;
    Query::Context context(pool, *reader);
    auto& segment = reader->segments()[0];
    auto* supplier = query.createWeight(context, 0)->scorerSupplier(pool, segment);
    Query::ScorerSupplier::BulkScorerContext bulkContext;
    std::vector<Query::ScorerSupplier*> filters;
    int64_t filterCost = INT64_MAX;
    for (auto* filter : enclosing) {
      auto* child = filter->createWeight(context, 0)->scorerSupplier(pool, segment);
      filters.push_back(child);
      filterCost = std::min(filterCost, child->cost());
    }
    if (!filters.empty()) bulkContext = {filterCost, filters, true};
    auto plan = supplier->planBulk(use, bulkContext);
    EXPECT_EQ(Answer::YES, plan.available);
    if (plan.available != Answer::YES) return {};
    EXPECT_EQ(FilterConsumption::NO, plan.windowFilterConsumption);
    auto* bulk = supplier->buildBulk(pool, plan);
    EXPECT_NE(nullptr, bulk);
    if (bulk == nullptr) return {};
    if (reposition) {
      EXPECT_EQ(Answer::YES, plan.supportsReposition());
      ScoreWindow ignored;
      bulk->matchNextWindow(ignored, nullptr, 0, N / 2);
      bulk->prepareMembership();
      bulk->reposition(0);
    }
    std::vector<int32_t> result;
    DocSetBuilder builder(N);
    int64_t count = 0;
    for (int32_t cursor = min; cursor < max;) {
      int32_t next;
      if (use == Use::COUNT_WINDOWS) {
        next = bulk->countNextWindow(count, &builder, domain, cursor, max);
      } else {
        ScoreWindow out;
        next = bulk->matchNextWindow(out, domain, cursor, max);
        result.insert(result.end(), out.docs.begin(), out.docs.begin() + out.size);
      }
      EXPECT_GT(next, cursor);
      if (next <= cursor) break;
      cursor = next;
    }
    if (use == Use::COUNT_WINDOWS) {
      auto docs = builder.build();
      EXPECT_EQ(count, docs->card());
      for (int32_t doc = 0; doc < N; doc++) if (docs->get(doc)) result.push_back(doc);
    }
    return result;
  }
};

TEST_F(RangeFilterTest, ForcedConjunctionPlansPreserveMembership) {
  TermQuery a("body_w", "threshold"), b("body_w", "b");
  NumericPredicateQuery range("proof_i", 0, N - 1);
  std::array<Query*, 3> required{&a, &b, &range};
  BooleanQuery query{std::span(required).first(2), {}, {}, std::span(required).last(1)};
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 768) {
    if (possible(doc)) expected.push_back(doc);
  }
  int64_t windows[2]{};
  [[maybe_unused]] int64_t verified[2]{};
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    for (int32_t lead = 0; lead < 3; lead++) {
      std::array<size_t, 3> order{0, 1, 2};
      std::rotate(order.begin(), order.begin() + lead, order.begin() + lead + 1);
      for (int32_t mask = 0; mask < 8; mask++) {
        if ((mask & (1 << lead)) != 0) continue;
        for (bool proofs : {false, true}) {
          std::array<Consumption, 3> consumption;
          for (size_t i = 0; i < 3; i++) {
            consumption[i] = (mask & (1 << i)) != 0
                ? Consumption::CANDIDATES : Consumption::WINDOWS;
          }
          std::array<bool, 3> observed{};
          std::array<std::vector<size_t>, 2> stages;
          BooleanQuery::ConjunctionControlForTests control{order, consumption, proofs,
              [&](size_t ordinal, Consumption stage, size_t position, Query::ScorerSupplier&,
                  const Query::PlanContext&, const Query::ScorerShape& shape) {
                ASSERT_LT(ordinal, 3u);
                EXPECT_FALSE(observed[ordinal]);
                observed[ordinal] = true;
                bool candidate = stage == Consumption::CANDIDATES;
                bool isLead = !candidate && position == 0;
                EXPECT_EQ(position, stages[candidate].size());
                stages[candidate].push_back(ordinal);
                EXPECT_EQ(ordinal == (size_t)lead, isLead);
                EXPECT_EQ(consumption[ordinal], stage);
                EXPECT_EQ(stage, shape.membershipConsumption);
              }};
          ConjunctionPlanControl force(control);
          SkipStatsScope stats;
          int64_t calls = 0;
          NumericVerificationObserver observe([&](int32_t) { calls++; });
          EXPECT_EQ(expected, membership(query, use));
          EXPECT_TRUE(std::ranges::all_of(observed, [](bool seen) { return seen; }));
          std::array<std::vector<size_t>, 2> expectedStages;
          for (size_t ordinal : order) {
            expectedStages[consumption[ordinal] == Consumption::CANDIDATES].push_back(ordinal);
          }
          EXPECT_EQ(expectedStages, stages);
          if (use == Use::COUNT_WINDOWS && lead == 0 && mask == 4) {
            windows[proofs] = SkipStats::conjDenseCountWindows;
            verified[proofs] = calls;
          }
        }
      }
    }
  }
  EXPECT_LT(windows[1], windows[0]);
#ifndef NDEBUG
  EXPECT_LT(verified[1], verified[0]);
#endif
  EXPECT_EQ(nullptr, BooleanQuery::conjunctionControlForTests);
}

TEST_F(RangeFilterTest, ForcedCandidateOrderPreservesStages) {
  TermQuery a("body_w", "threshold"), b("body_w", "b");
  NumericPredicateQuery first("proof_i", 0, N - 1), second("proof2_i", 0, N - 1);
  std::array<Query*, 4> required{&a, &b, &first, &second};
  BooleanQuery query{std::span(required).first(3), {}, {}, std::span(required).last(1)};
  std::array<size_t, 4> order{0, 2, 1, 3};
  std::array<Consumption, 4> consumption{Consumption::WINDOWS, Consumption::WINDOWS,
      Consumption::CANDIDATES, Consumption::CANDIDATES};
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 768) {
    if (possible(doc) && doc >= 8192 && doc < 20480) expected.push_back(doc);
  }
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    std::array<std::vector<size_t>, 2> stages;
    BooleanQuery::ConjunctionControlForTests control{order, consumption, true,
        [&](size_t ordinal, Consumption stage, size_t position, Query::ScorerSupplier&,
            const Query::PlanContext&, const Query::ScorerShape& shape) {
          EXPECT_EQ(stage, shape.membershipConsumption);
          auto& sequence = stages[stage == Consumption::CANDIDATES];
          EXPECT_EQ(position, sequence.size());
          sequence.push_back(ordinal);
        }};
    ConjunctionPlanControl force(control);
    EXPECT_EQ(expected, membership(query, use));
    EXPECT_EQ((std::vector<size_t>{0, 1}), stages[0]);
    EXPECT_EQ((std::vector<size_t>{2, 3}), stages[1]);
  }
  BooleanQuery::ConjunctionControlForTests invalid{order, consumption, true, {}};
  order[3] = 2;
  EXPECT_THROW({ ConjunctionPlanControl force(invalid); }, std::invalid_argument);
  order[3] = 4;
  EXPECT_THROW({ ConjunctionPlanControl force(invalid); }, std::invalid_argument);
  order[3] = 3;
  consumption[0] = Consumption::CANDIDATES;
  EXPECT_THROW({ ConjunctionPlanControl force(invalid); }, std::invalid_argument);
  EXPECT_EQ(nullptr, BooleanQuery::conjunctionControlForTests);
}

TEST_F(RangeFilterTest, CountAndMatchVerifyOnlyExactSurvivors) {
  TermQuery a("body_w", "a"), b("body_w", "b");
  NumericPredicateQuery range("column_i", 0, 59000);
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range};
  BooleanQuery query(required, {}, {}, filters);
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 192) {
    if (2 * ((doc * 1543) % N) <= 59000) expected.push_back(doc);
  }
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    int64_t verified = 0;
    NumericVerificationObserver observe([&](int32_t doc) {
      verified++;
      EXPECT_EQ(0, doc % 192);
    });
    EXPECT_EQ(expected, membership(query, use));
    EXPECT_LE(verified, N / 192 + 1);
#ifndef NDEBUG
    EXPECT_GT(verified, 0);
#endif
  }
}

TEST_F(RangeFilterTest, CandidateFiltersFollowExclusionsAndDomain) {
  TermQuery a("body_w", "a"), b("body_w", "b"), excluded("body_w", "excluded");
  NumericPredicateQuery range("column_i", 0, 59000);
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range}, prohibited{&excluded};
  BooleanQuery positive(required, {}, {}, filters), negative(required, {}, prohibited, filters);
  RAMBitDocSet domain(N);
  domain.mutableBits().set(1);
  int64_t verified = 0;
  NumericVerificationObserver observe([&](int32_t) { verified++; });
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    EXPECT_TRUE(membership(positive, use, &domain).empty());
    EXPECT_TRUE(membership(negative, use).empty());
  }
  EXPECT_EQ(0, verified);
}

TEST_F(RangeFilterTest, SparseCandidateFiltersFollowDomain) {
  TermQuery a("body_w", "rare"), b("body_w", "b");
  NumericPredicateQuery range("column_i", 0, 59000);
  std::array<Query*, 2> required{&a, &b};
  BooleanQuery body(required, {}, {}, {});
  std::array<Query*, 1> filters{&range};
  RAMBitDocSet domain(N);
  domain.mutableBits().set(1);
  int64_t verified = 0;
  NumericVerificationObserver observe([&](int32_t) { verified++; });
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    EXPECT_TRUE(membership(body, use, &domain, filters).empty());
  }
  EXPECT_EQ(0, verified);
}

TEST_F(RangeFilterTest, LeadOnlyAndMixedThresholdPlansResetWithFilters) {
  TermQuery a("body_w", "threshold"), b("body_w", "b");
  NumericPredicateQuery range("column_i", 0, 59000);
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range};
  for (size_t size : {1u, 2u}) {
    BooleanQuery query(std::span<Query*>(required).first(size), {}, {}, filters);
    for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
      SkipStatsScope stats;
      auto expected = membership(query, use);
      EXPECT_FALSE(expected.empty());
      EXPECT_EQ(expected, membership(query, use, nullptr, {}, true));
      EXPECT_GT(SkipStats::conjDenseCountWindows + SkipStats::conjDenseMatchWindows, 0);
    }
  }
}

TEST_F(RangeFilterTest, NumericSelectionPricesDenseAndSettledProduction) {
  // Scanning a random column costs about as much as verifying most docs.
  // Proofs keep settled-zone skips when sparse candidates are cheaper.
  // The cheap quote charges crossing leaves even if later rank refinement
  // can settle them, so the multivalued range also verifies sparse candidates.
  struct Case {
    const char* field;
    int64_t candidates;
    bool verifies;
  };
  for (auto [field, candidates, verifies] : {Case{"column_i", 64, true},
           Case{"column_i", N * 3 / 10, true}, Case{"column_i", N * 9 / 10, false},
           Case{"cluster_i", 64, true}, Case{"cluster_i", N * 3 / 10, false},
           Case{"cluster_i", N * 9 / 10, false}, Case{"cluster_is", 64, true},
           Case{"cluster_is", N * 3 / 10, false}}) {
    SCOPED_TRACE(::testing::Message() << field << " " << candidates);
    NumericPredicateQuery range(field, 4096, 28671);
    MemPool pool;
    Query::Context context(pool, *reader);
    auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(
        candidates, N, Use::COUNT_WINDOWS));
    build.membershipConsumption = Consumption::EITHER;
    auto shape = supplier->describeScorer(build);
    auto* plan = supplier->resolve(pool, build);
    EXPECT_EQ(shape.membershipConsumption, plan->shape().membershipConsumption);
    if (std::string_view(field) == "cluster_is") {
      EXPECT_EQ(nullptr, plan->buildMembershipProof(pool));
    }
    EXPECT_EQ(verifies ? Consumption::CANDIDATES : Consumption::WINDOWS,
              shape.membershipConsumption);
    auto* scorer = plan->build(pool);
    if (verifies) {
      EXPECT_NE(nullptr, dynamic_cast<NumericPredicateQuery::PredicateScorer<
          IntColReader::SparseIterator>*>(scorer));
    } else if (std::string_view(field).starts_with("cluster_")) {
      EXPECT_NE(nullptr, dynamic_cast<NumericPredicateQuery::ZoneMapScorer*>(scorer));
    } else {
      EXPECT_NE(nullptr, dynamic_cast<NumericPredicateQuery::PredicateScorer<
          IntColReader::Iterator>*>(scorer));
    }
  }
}

TEST_F(RangeFilterTest, ProofsSkipProductionInDenseAndSparseMembership) {
  NumericPredicateQuery first("proof_i", 0, N - 1), second("proof2_i", 0, N - 1);
  TermQuery b("body_w", "b");
  std::array<Query*, 2> filters{&first, &second};
  for (auto [lead, step] : {std::pair{"threshold", 256}, {"rare", 2048}, {"dense_a", 16}}) {
    TermQuery a("body_w", lead);
    std::array<Query*, 2> required{&a, &b};
    BooleanQuery body(required, {}, {}, {});
    std::vector<int32_t> expected;
    for (int32_t doc = 0; doc < N; doc += step) {
      if (doc % 3 == 0 && possible(doc) && doc >= 8192 && doc < 20480) expected.push_back(doc);
    }
    for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
      SkipStatsScope stats;
      int64_t verified = 0;
      NumericVerificationObserver observe([&](int32_t doc) {
        EXPECT_TRUE(possible(doc) && doc >= 8192 && doc < 20480);
        verified++;
      });
      EXPECT_EQ(expected, membership(body, use, nullptr, filters));
      EXPECT_LE(verified, (int64_t)expected.size() * 2);
      EXPECT_LT(SkipStats::countBulkFillDocs, N / 3);
      if (step == 2048) {
        EXPECT_EQ(0, SkipStats::conjDenseCountWindows + SkipStats::conjDenseMatchWindows);
        EXPECT_LT(SkipStats::advanceCalls, 2 * N / step);
      } else {
        EXPECT_GT(SkipStats::conjDenseCountWindows + SkipStats::conjDenseMatchWindows, 0);
        EXPECT_LE(SkipStats::conjDenseCountWindows + SkipStats::conjDenseMatchWindows, 4);
      }
    }
  }
}

TEST_F(RangeFilterTest, ProofsRejectInsideWindowsAfterExactElimination) {
  TermQuery a("body_w", "threshold"), b("body_w", "b"), excluded("body_w", "excluded");
  NumericPredicateQuery range("striped_i", 0, 0);
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range}, prohibited{&excluded};
  BooleanQuery query(required, {}, {}, filters), negative(required, {}, prohibited, filters);
  RAMBitDocSet domain(N);
  domain.mutableBits().set(768);
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 768) {
    if ((doc / 512) % 2 == 0) expected.push_back(doc);
  }
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    int64_t verified = 0;
    NumericVerificationObserver observe([&](int32_t doc) {
      EXPECT_EQ(0, (doc / 512) % 2);
      EXPECT_EQ(0, doc % 768);
      verified++;
    });
    EXPECT_EQ(expected, membership(query, use));
#ifndef NDEBUG
    EXPECT_EQ((int64_t)expected.size(), verified);
#endif
    verified = 0;
    EXPECT_TRUE(membership(query, use, &domain).empty());
    EXPECT_TRUE(membership(negative, use).empty());
    EXPECT_EQ(0, verified);
  }
}

TEST_F(RangeFilterTest, PartlySettledScanSharesProofsWithCandidateConsumption) {
  NumericPredicateQuery range("partial_i", 0, N - 1);
  MemPool pool;
  Query::Context context(pool, *reader);
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(N, N, Use::COUNT_WINDOWS));
  build.membershipConsumption = Consumption::WINDOWS;
  auto* plan = supplier->resolve(pool, build);
  EXPECT_NE(nullptr, dynamic_cast<NumericPredicateQuery::PredicateScorer<
      IntColReader::Iterator>*>(plan->build(pool)));
  auto* proof = plan->buildMembershipProof(pool);
  ASSERT_NE(nullptr, proof);
  EXPECT_EQ(N / 4, proof->nextPossible(0));

  TermQuery a("body_w", "threshold"), b("body_w", "b");
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range};
  BooleanQuery query(required, {}, {}, filters);
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 768) {
    if (doc >= N / 4 && 2 * ((doc * 1543) % N) < N) expected.push_back(doc);
  }
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    SkipStatsScope stats;
    NumericVerificationObserver observe([&](int32_t doc) { EXPECT_GE(doc, N / 4); });
    EXPECT_EQ(expected, membership(query, use));
    EXPECT_EQ(1, SkipStats::numericPredicateSparseVerifyArms);
    EXPECT_EQ(6, SkipStats::conjDenseCountWindows + SkipStats::conjDenseMatchWindows);
    EXPECT_LT(SkipStats::countBulkFillDocs, N / 3);
  }
}

TEST_F(RangeFilterTest, NumericPlansRefineOnlyWhenConsumed) {
  NumericPredicateQuery range("striped_i", 0, 0);
  MemPool pool;
  Query::Context context(pool, *reader);
  SkipStatsScope stats;
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  int64_t cost = supplier->cost();
  auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(1, N, Use::COUNT_WINDOWS));
  build.membershipConsumption = Consumption::CANDIDATES;
  auto* plan = supplier->resolve(pool, build);
  auto* scorer = plan->build(pool);
  auto* proof = plan->buildMembershipProof(pool);
  ASSERT_NE(nullptr, scorer);
  ASSERT_NE(nullptr, proof);
  EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
  EXPECT_EQ(0, SkipStats::numericPredicateProofIndexes);
  EXPECT_EQ(0, proof->nextPossible(0));
  EXPECT_EQ(1024, proof->nextPossible(512));
  EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
  EXPECT_EQ(0, SkipStats::numericPredicateProofIndexes);
  proof->reset();
  EXPECT_EQ(0, proof->nextPossible(0));

  build.demand = Query::Demand::fromCandidatesAndSpan(N, N, Use::COUNT_WINDOWS);
  auto limited = build;
  limited.expectedSpan = 64;
  auto* local = supplier->resolve(pool, limited)->buildMembershipProof(pool);
  ASSERT_NE(nullptr, local);
  EXPECT_EQ(1024, local->nextPossible(512));
  EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
  EXPECT_EQ(0, SkipStats::numericPredicateProofIndexes);
  auto* full = supplier->resolve(pool, build)->buildMembershipProof(pool);
  ASSERT_NE(nullptr, full);
  EXPECT_EQ(0, SkipStats::numericPredicateProofIndexes);
  EXPECT_EQ(1024, full->nextPossible(512));
  EXPECT_EQ(1, SkipStats::numericPredicateProofIndexes);
  EXPECT_EQ(N / IntColReader::BLOCK_SIZE, SkipStats::numericPredicateClassifiedBlocks);
  EXPECT_EQ(cost, supplier->cost());
  full->reset();
  EXPECT_EQ(0, full->nextPossible(0));
  EXPECT_EQ(1, SkipStats::numericPredicateProofIndexes);
}

TEST_F(RangeFilterTest, PointsFactsAndColumnProofsFollowResolvedConsumption) {
  NumericPredicateQuery range("proof_points_i", 0, N - 1);
  for (auto [candidates, consumption] : {
           std::pair{1, Consumption::EITHER}, {N, Consumption::EITHER},
           {N, Consumption::CANDIDATES}}) {
    MemPool pool;
    Query::Context context(pool, *reader);
    SkipStatsScope stats;
    auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    EXPECT_EQ(N / 2, supplier->cost());
    auto build = supplier->makePlanContext(
        Query::Demand::fromCandidatesAndSpan(candidates, N, Use::COUNT_WINDOWS));
    build.membershipConsumption = consumption;
    auto shape = supplier->describeScorer(build);
    EXPECT_EQ(Query::MatchState::NONEMPTY, shape.matchState);
    auto* plan = supplier->resolve(pool, build);
    auto* scorer = plan->build(pool);
    auto* proof = plan->buildMembershipProof(pool);
    EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
    if (consumption == Consumption::CANDIDATES) {
      EXPECT_EQ(nullptr, proof);
      EXPECT_EQ(1, SkipStats::numericPredicatePointsArms);
    } else {
      ASSERT_NE(nullptr, proof);
      EXPECT_EQ(4096, proof->nextPossible(0));
      EXPECT_EQ(candidates == 1 ? 0 : 1, SkipStats::numericPredicateProofIndexes);
      EXPECT_EQ(candidates == 1 ? 1 : 0, SkipStats::numericPredicateSparseVerifyArms);
      EXPECT_EQ(candidates == N ? 1 : 0, SkipStats::numericPredicateZoneArms);
    }
    std::array<uint64_t, 64> bits;
    bits.fill(~0ULL);
    scorer->intersectCandidateBits(bits, 2048);
    for (int32_t i = 0; i < 4096; i++) {
      EXPECT_EQ(possible(2048 + i), (bits[(size_t)(i >> 6)] & (1ULL << (i & 63))) != 0);
    }
  }
  MemPool pool;
  Query::Context context(pool, *reader);
  SkipStatsScope stats;
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  Query::ScorerSupplier::BulkScorerContext bulk;
  bulk.requireConstantCount = true;
  auto plan = supplier->planBulk(Use::COUNT_WINDOWS, bulk);
  EXPECT_TRUE(plan.hasConstantCount());
  EXPECT_EQ(N / 2, plan.constantCount);
  EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
  EXPECT_EQ(0, SkipStats::numericPredicatePointsArms);
  EXPECT_EQ(0, SkipStats::numericPredicateZoneArms);
}

TEST_F(RangeFilterTest, SparseColumnProofsRespectBoundsAndReset) {
  TermQuery a("body_w", "threshold"), b("body_w", "b");
  NumericPredicateQuery range("holes_i", 0, N - 1);
  std::array<Query*, 2> required{&a, &b};
  std::array<Query*, 1> filters{&range};
  BooleanQuery query(required, {}, {}, filters);
  for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
    for (auto [min, max] : {std::pair{0, N}, {6500, 23000}, {24576, N}, {0, 4096}}) {
      std::vector<int32_t> expected;
      for (int32_t doc = min; doc < max; doc++) {
        if (doc % 768 == 0 && doc % 5 != 0 && possible(doc)) expected.push_back(doc);
      }
      NumericVerificationObserver observe([&](int32_t doc) {
        EXPECT_GE(doc, min);
        EXPECT_LT(doc, max);
      });
      EXPECT_EQ(expected, membership(query, use, nullptr, {}, false, min, max));
    }
    auto expected = membership(query, use);
    EXPECT_EQ(expected, membership(query, use, nullptr, {}, true));
  }
}

TEST_F(RangeFilterTest, SparseProofRetainsMissingBoundaryGaps) {
  NumericPredicateQuery range("edge_i", 0, N - 1);
  MemPool pool;
  Query::Context context(pool, *reader);
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  auto* plan = supplier->resolve(pool, supplier->makePlanContext(
      Query::Demand::fromCandidatesAndSpan(N, N, Use::COUNT_WINDOWS)));
  auto* proof = plan->buildMembershipProof(pool);
  ASSERT_NE(nullptr, proof);
  NumericVerificationObserver observe([](int32_t) { FAIL() << "proof decoded a value"; });
  EXPECT_EQ(2048, proof->nextPossible(0));
  EXPECT_EQ(PostingsReader::END, proof->nextPossible(N - 2048));
  proof->reset();
  EXPECT_EQ(2048, proof->nextPossible(0));
}

TEST_F(LuxirTest, RangeProofIndexSpansBitmapWords) {
  constexpr int32_t N = 65 * 4096 + 17;
  auto possible = [](int32_t doc) {
    return doc >= 512 && doc < 65 * 4096
        && (doc < 63 * 4096 + 512 || doc >= 64 * 4096 + 512);
  };
  CollectionHelper helper;
  auto writer = helper.getIndexWriter();
  auto& inverter = writer->obtainInverter();
  auto& field = inverter.getIndexHandler("proof_i");
  for (int32_t doc = 0; doc < N; doc++) {
    inverter.startDoc();
    field.index(inverter, (int64_t)(possible(doc) ? 0 : 1));
    inverter.finishDoc();
  }
  writer->releaseInverter(inverter, true);
  helper.commit();
  IndexReader reader(writer->dir);
  ASSERT_EQ(1u, reader.segments().size());
  MemPool pool;
  Query::Context context(pool, reader);
  NumericPredicateQuery range("proof_i", 0, 0);
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader.segments()[0]);
  auto* plan = supplier->resolve(pool, supplier->makePlanContext(
      Query::Demand::fromCandidatesAndSpan(N, N, Use::COUNT_WINDOWS)));
  auto* proof = plan->buildMembershipProof(pool);
  ASSERT_NE(nullptr, proof);
  NumericVerificationObserver observe([](int32_t) { FAIL() << "proof decoded a value"; });
  for (int32_t base : {0, 62 * 4096, 63 * 4096, 64 * 4096, 65 * 4096, 512}) {
    int32_t expected = base;
    while (expected < N && !possible(expected)) expected++;
    EXPECT_EQ(expected == N ? PostingsReader::END : expected, proof->nextPossible(base));
    std::array<uint64_t, 64> bits;
    bits.fill(~0ULL);
    proof->intersect(bits, base);
    for (int32_t i = 0; i < 4096; i++) {
      EXPECT_EQ(possible(base + i), (bits[(size_t)(i >> 6)] & (1ULL << (i & 63))) != 0);
    }
  }
  proof->reset();
  EXPECT_EQ(512, proof->nextPossible(0));
  EXPECT_EQ(63 * 4096 + 512, proof->possibleEnd());
}

TEST_F(RangeFilterTest, ProofResourcesAreIndependentOfScorerCursors) {
  for (auto field : {"proof_i", "holes_i"}) {
    for (int64_t candidates : {1, N}) {
      for (auto consumption : {Consumption::WINDOWS, Consumption::CANDIDATES}) {
        NumericPredicateQuery range(field, 0, N - 1);
        MemPool pool;
        Query::Context context(pool, *reader);
        auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
        auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(candidates, N, Use::COUNT_WINDOWS));
        build.membershipConsumption = consumption;
        auto* plan = supplier->resolve(pool, build);
        auto* scorer = plan->build(pool);
        auto* proof = plan->buildMembershipProof(pool);
        ASSERT_NE(nullptr, proof);
        {
          NumericVerificationObserver observe([](int32_t) { FAIL() << "proof decoded a value"; });
          for (int32_t target : {0, 4095, 4096, 4609, 5001, 12287, 12288, 16383, 24576, N, 0}) {
            int32_t bound = proof->nextPossible(target);
            EXPECT_GE(bound, target);
            for (int32_t doc = target; doc < std::min(bound, N); doc++) {
              EXPECT_FALSE(possible(doc) && (std::string_view(field) != "holes_i" || doc % 5 != 0));
            }
          }
          for (int32_t base : {4001, 4050, 4095, 12287, 24576, 4096, 5001}) {
            std::array<uint64_t, 64> bits;
            bits.fill(~0ULL);
            proof->intersect(bits, base);
            for (int32_t bit = 0; bit < 4096; bit++) {
              bool kept = (bits[(size_t)(bit >> 6)] & (1ULL << (bit & 63))) != 0;
              if (std::string_view(field) == "proof_i") {
                EXPECT_EQ(possible(base + bit), kept);
              } else if (possible(base + bit) && (base + bit) % 5 != 0) {
                EXPECT_TRUE(kept);
              }
            }
          }
          EXPECT_EQ(-1, scorer->docId());
          proof->reset();
          EXPECT_GT(proof->nextPossible(0), 0);
        }
        EXPECT_EQ(4096, scorer->advanceWindow(0, N));
      }
    }
  }
  for (auto field : {"column_i", "points_i", "cluster_is"}) {
    NumericPredicateQuery range(field, 0, 65535);
    MemPool pool;
    Query::Context context(pool, *reader);
    auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    auto* plan = supplier->resolve(pool, supplier->makePlanContext(Query::Demand::fromLeadCost(N)));
    EXPECT_EQ(nullptr, plan->buildMembershipProof(pool));
  }
}

TEST_F(RangeFilterTest, NestedRangesRetainPointsMissingAndMultivalueSemantics) {
  TermQuery a("body_w", "a"), b("body_w", "b");
  std::array<Query*, 2> terms{&a, &b};
  BooleanQuery body(terms, {}, {}, {});
  std::array<Query*, 1> required{&body};
  for (auto field : {"column_i", "points_i", "multi_is"}) {
    for (auto [lo, hi] : {std::pair<int64_t, int64_t>{0, 2047}, {30000, 59000}}) {
      NumericPredicateQuery range(field, lo, hi);
      std::array<Query*, 1> filters{&range};
      BooleanQuery query(required, {}, {}, filters);
      std::vector<int32_t> expected;
      for (int32_t doc = 0; doc < N; doc += 192) {
        int64_t value = 2 * ((doc * 1543) % N);
        bool matches = lo <= value && value <= hi;
        if (std::string_view(field) == "multi_is") {
          matches = doc % 5 != 0 && (matches
              || (lo <= value + 1000 && value + 1000 <= hi));
        }
        if (matches) expected.push_back(doc);
      }
      for (auto use : {Use::COUNT_WINDOWS, Use::MATCH_WINDOWS}) {
        EXPECT_EQ(expected, membership(query, use));
      }
    }
  }
}

TEST_F(RangeFilterTest, EmptyNestedRangesReturnNoScoredMatches) {
  TermQuery a("body_w", "a"), b("body_w", "b");
  std::array<Query*, 2> terms{&a, &b};
  BooleanQuery body(terms, {}, {}, {});
  std::array<Query*, 1> required{&body};
  for (auto field : {"column_i", "points_i"}) {
    NumericPredicateQuery range(field, 1, 1);
    std::array<Query*, 1> filters{&range};
    BooleanQuery query(required, {}, {}, filters);
    MemPool pool;
    Query::Context context(pool, *reader);
    auto* supplier = query.createWeight(context, Query::NEED_SCORES)
        ->scorerSupplier(pool, reader->segments()[0]);
    if (supplier == nullptr) continue;
    auto* bulk = supplier->bulkScorer(pool);
    TopDocsCollector collector(10);
    if (bulk != nullptr) {
      collectTopKWindowed(0, bulk, nullptr, collector, nullptr, N, true);
    } else if (auto* scorer = buildScorerForTests(pool, *supplier, INT64_MAX)) {
      collectTopK(0, scorer, nullptr, nullptr, collector, true);
    }
    EXPECT_EQ(0, collector.totalHits());
  }
}

TEST_F(RangeFilterTest, ScoredConsumersHonorDeclaredFilterConsumption) {
  TermQuery a("body_w", "a"), b("body_w", "c");
  TermQuery da("body_w", "dense_a"), db("body_w", "dense_b");
  NumericPredicateQuery range("column_i", 0, 59000);
  NumericPredicateQuery points("points_i", 0, 59000);
  std::array<Query*, 1> required{&a}, optional{&b}, filters{&range};
  std::array<Query*, 1> knownFilters{&points};
  std::array<Query*, 2> disjunction{&a, &b}, conjunction{&da, &db};
  BooleanQuery term(required, {}, {}, filters), mandopt(required, optional, {}, knownFilters);
  BooleanQuery maxscore({}, disjunction, {}, knownFilters, 1), dense(conjunction, {}, {}, filters);
  for (Query* query : {&term, &dense, &mandopt, &maxscore}) {
    std::vector<TopDocsCollector::ScoreDoc> expected;
    for (bool pull : {true, false}) {
      MemPool pool;
      Query::Context context(pool, *reader);
      auto* supplier = query->createWeight(context, Query::NEED_SCORES | Query::ALLOW_PRUNING)
          ->scorerSupplier(pool, reader->segments()[0]);
      TopDocsCollector collector(1000);
      SkipStatsScope stats;
      if (pull) {
        auto* scorer = buildScorerForTests(pool, *supplier, INT64_MAX);
        collectTopK(0, scorer, nullptr, nullptr, collector, true);
        auto hits = collector.sort();
        expected.assign(hits.begin(), hits.end());
      } else {
        auto plan = supplier->planBulk(Use::SCORED_WINDOWS, {});
        ASSERT_EQ(Answer::YES, plan.available);
        bool probes = query == &term || query == &dense;
        EXPECT_EQ(probes ? FilterConsumption::FILL_OR_PROBE : FilterConsumption::FILL,
                  plan.windowFilterConsumption);
        auto* bulk = supplier->buildBulk(pool, plan);
        ASSERT_NE(nullptr, bulk);
        int64_t verified = 0;
        NumericVerificationObserver observe([&](int32_t) { verified++; });
        collectTopKWindowed(0, bulk, nullptr, collector, nullptr, N, true);
        assertTopKEquivalent(expected, collector.sort());
        if (probes) {
          EXPECT_LE(verified, 1024);
        } else {
          EXPECT_EQ(0, SkipStats::numericPredicateSparseVerifyArms);
        }
        if (query == &dense) {
          EXPECT_GT(SkipStats::conjDenseScoredWindows, 0);
        }
      }
    }
  }
}

TEST_F(RangeFilterTest, FillOnlyBodiesKeepUnknownFiltersOnPull) {
  TermQuery a("body_w", "a"), b("body_w", "c");
  NumericPredicateQuery range("column_i", 0, 59000);
  std::array<Query*, 1> required{&a}, optional{&b}, filters{&range};
  std::array<Query*, 2> disjunction{&a, &b};
  BooleanQuery mandopt(required, optional, {}, {}), maxscore({}, disjunction, {}, {}, 1);
  for (Query* body : {&mandopt, &maxscore}) {
    std::array<Query*, 1> bodies{body};
    BooleanQuery query(bodies, {}, {}, filters);
    MemPool pool;
    Query::Context context(pool, *reader);
    auto* supplier = query.createWeight(context, Query::NEED_SCORES | Query::ALLOW_PRUNING)
        ->scorerSupplier(pool, reader->segments()[0]);
    EXPECT_EQ(Answer::NO, supplier->planBulk(Use::SCORED_WINDOWS, {}).available);
    auto* scorer = buildScorerForTests(pool, *supplier, INT64_MAX);
    int32_t count = 0, expected = 0;
    while (scorer->next() != PostingsReader::END) count++;
    for (int32_t doc = 0; doc < N; doc++) {
      bool positive = doc % 64 == 0 || (body == &maxscore && doc % 64 == 1);
      expected += positive && 2 * ((doc * 1543) % N) <= 59000;
    }
    EXPECT_EQ(expected, count);
  }
}

TEST_F(RangeFilterTest, ProbePreparationDoesNotFillAndObserversRestore) {
  NumericPredicateQuery range("column_i", 0, 59000);
  MemPool pool;
  Query::Context context(pool, *reader);
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(2, N));
  build.membershipConsumption = Consumption::CANDIDATES;
  auto* plan = supplier->resolve(pool, build);
  ASSERT_EQ(Consumption::CANDIDATES, plan->shape().membershipConsumption);
  std::array<Query::Scorer*, 1> scorers{plan->build(pool)};
  WindowFilter filter(pool, scorers, true);
  int64_t verified = 0;
  NumericVerificationObserver outer([&](int32_t) { verified++; });
  EXPECT_GT(filter.prepare(0, 128), 0);
  EXPECT_EQ(0, verified);
  {
    NumericVerificationObserver inner([](int32_t) { FAIL() << "empty mask verified"; });
    std::array<uint64_t, 2> empty{};
    filter.intersect(empty, 0);
  }
  std::array<uint64_t, 2> bits{1, 1};
  filter.intersect(bits, 0);
#ifndef NDEBUG
  EXPECT_EQ(2, verified);
#endif
}

} // namespace

TEST_F(RangeFilterTest, PartialExposureClassifiesTouchedBlocksAndCanStillExhaust) {
  for (auto field : {"proof_points_i", "holes_i", "cluster_is"}) {
    NumericPredicateQuery range(field, 0, N - 1);
    MemPool pool;
    Query::Context context(pool, *reader);
    SkipStatsScope stats;
    auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    Query::ScorerSupplier::BulkScorerContext bulkContext;
    bulkContext.expectedConsumption = Query::Demand::fromCandidatesAndSpan(64, 64, Use::MATCH_WINDOWS);
    auto plan = supplier->planBulk(Use::MATCH_WINDOWS, bulkContext);
    ASSERT_EQ(Answer::YES, plan.available);
    ASSERT_EQ(Answer::YES, plan.supportsReposition());
    auto* bulk = supplier->buildBulk(pool, plan);
    EXPECT_EQ(0, SkipStats::numericPredicateClassifiedBlocks);
    ScoreWindow out;
    bulk->matchNextWindow(out, nullptr, 4096, 4160);
    EXPECT_EQ(std::string_view(field) == "holes_i" ? 52 : 64, out.size);
    EXPECT_EQ(1, SkipStats::numericPredicateClassifiedBlocks);
    bulk->prepareMembership();
    bulk->reposition(4096);
    bulk->matchNextWindow(out, nullptr, 4096, 4160);
    EXPECT_EQ(1, SkipStats::numericPredicateClassifiedBlocks);

    // The end sentinel maps to the final sparse/multivalued value rank.
    bulk->reposition(N - 64);
    bulk->matchNextWindow(out, nullptr, N - 64, N);
    EXPECT_EQ(std::string_view(field) == "cluster_is" ? 64 : 0, out.size);
    EXPECT_EQ(2, SkipStats::numericPredicateClassifiedBlocks);

    // A broad consumer fills missing plans, sharing those built on touch.
    auto eager = supplier->planBulk(Use::MATCH_WINDOWS, {});
    supplier->buildBulk(pool, eager)->matchNextWindow(out, nullptr, 4096, 4160);
    int64_t values = std::string_view(field) == "cluster_is" ? 2 * N
        : std::string_view(field) == "holes_i" ? N - (N + 4) / 5 : N;
    int64_t blocks = (values + IntColReader::BLOCK_SIZE - 1) / IntColReader::BLOCK_SIZE;
    EXPECT_EQ(blocks, SkipStats::numericPredicateClassifiedBlocks);

    // A deliberately wrong short estimate must not bound execution, even
    // after repositioning. Includes sparse ranks and a multivalued column.
    bulk->reposition(0);
    std::vector<int32_t> actual, expected;
    for (int32_t cursor = 0; cursor < N;) {
      int32_t next = bulk->matchNextWindow(out, nullptr, cursor, N);
      actual.insert(actual.end(), out.docs.begin(), out.docs.begin() + out.size);
      ASSERT_GT(next, cursor);
      cursor = next;
    }
    for (int32_t doc = 0; doc < N; doc++) {
      if (std::string_view(field) == "cluster_is"
          || (possible(doc) && (std::string_view(field) != "holes_i" || doc % 5 != 0))) {
        expected.push_back(doc);
      }
    }
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(blocks, SkipStats::numericPredicateClassifiedBlocks);
  }
}

TEST_F(RangeFilterTest, PartialExposureSurvivesNestedBooleanResolution) {
  NumericPredicateQuery first("proof_points_i", 0, N - 1), second("cluster_i", 4096, 8191);
  TermQuery term("body_w", "pad");
  std::array<Query*, 2> alternatives{&first, &second};
  BooleanQuery inner({}, alternatives, {}, {}, 1);
  std::array<Query*, 2> required{&term, &inner};
  BooleanQuery outer(required, {}, {}, {});
  for (Query* query : std::array<Query*, 2>{&inner, &outer}) {
    MemPool pool;
    Query::Context context(pool, *reader);
    SkipStatsScope stats;
    auto* supplier = query->createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(64, 64));
    build.expectedSpan = 64;
    auto* scorer = supplier->resolve(pool, build)->build(pool);
    std::array<uint64_t, 1> bits{};
    scorer->fillWindowBits(bits, 4096, 4160);
    EXPECT_EQ(~0ULL, bits[0]);
    EXPECT_LE(SkipStats::numericPredicateClassifiedBlocks, 2);
    EXPECT_EQ(0, SkipStats::numericPredicateProofIndexes);
  }
}

TEST_F(RangeFilterTest, CandidateExposurePricesOnlyTheIncomingFilterStream) {
  TermQuery term("body_w", "dense_a");
  NumericPredicateQuery range("proof_points_i", 0, N - 1);
  std::array<Query*, 1> required{&term}, filters{&range};
  BooleanQuery query(required, {}, {}, filters);
  std::vector<int32_t> expected;
  for (int32_t doc = 0; doc < N; doc += 16) if (possible(doc)) expected.push_back(doc);
  for (int64_t candidates : {1, N}) {
    MemPool pool;
    Query::Context context(pool, *reader);
    SkipStatsScope stats;
    auto* supplier = query.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    Query::ScorerSupplier::BulkScorerContext bulkContext;
    bulkContext.expectedConsumption = Query::Demand::fromCandidatesAndSpan(candidates, 4096, Use::MATCH_WINDOWS);
    auto plan = supplier->planBulk(Use::MATCH_WINDOWS, bulkContext);
    ASSERT_EQ(Answer::YES, plan.available);
    auto* bulk = supplier->buildBulk(pool, plan);
    EXPECT_EQ(candidates == 1 ? 1 : 0, SkipStats::numericPredicateSparseVerifyArms);
    EXPECT_EQ(candidates == 1 ? 0 : 1, SkipStats::numericPredicateZoneArms);
    std::vector<int32_t> actual;
    for (int32_t cursor = 0; cursor < N;) {
      ScoreWindow out;
      int32_t next = bulk->matchNextWindow(out, nullptr, cursor, N);
      actual.insert(actual.end(), out.docs.begin(), out.docs.begin() + out.size);
      ASSERT_GT(next, cursor);
      cursor = next;
    }
    EXPECT_EQ(expected, actual);
  }
}

TEST_F(RangeFilterTest, ExplicitWindowExposureComparesZonesBelowLegacyVerifierFence) {
  NumericPredicateQuery range("proof_points_i", 0, N - 1);
  for (auto [use, explicitSpan] : {
           std::pair{Use::MATCH_WINDOWS, true}, {Use::MATCH_WINDOWS, false}, {Use::PULL, true}}) {
    MemPool pool;
    Query::Context context(pool, *reader);
    SkipStatsScope stats;
    auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
    auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(1024, 1024, use));
    if (explicitSpan) build.expectedSpan = 1024;
    auto* scorer = supplier->resolve(pool, build)->build(pool);
    bool zones = explicitSpan && use == Use::MATCH_WINDOWS;
    EXPECT_EQ(zones ? 1 : 0, SkipStats::numericPredicateZoneArms);
    EXPECT_EQ(zones ? 0 : 1, SkipStats::numericPredicateSparseVerifyArms);
    std::array<uint64_t, 1> bits{};
    scorer->fillWindowBits(bits, 4096, 4160);
    EXPECT_EQ(~0ULL, bits[0]);
    EXPECT_EQ(zones ? 1 : 0, SkipStats::numericPredicateClassifiedBlocks);
  }
}

TEST_F(RangeFilterTest, CandidateDemandAloneDoesNotMakeZoneSetupIncremental) {
  NumericPredicateQuery range("striped_i", 0, 0);
  MemPool pool;
  Query::Context context(pool, *reader);
  SkipStatsScope stats;
  auto* supplier = range.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  auto build = supplier->makePlanContext(Query::Demand::fromCandidatesAndSpan(N / 2, N / 2));
  build.membershipConsumption = Consumption::EITHER;
  auto* scorer = supplier->resolve(pool, build)->build(pool);
  ASSERT_EQ(1, SkipStats::numericPredicateZoneArms);
  std::array<uint64_t, 1> bits{};
  scorer->fillWindowBits(bits, 0, 64);
  EXPECT_EQ(~0ULL, bits[0]);
  EXPECT_EQ(N / IntColReader::BLOCK_SIZE, SkipStats::numericPredicateClassifiedBlocks);
}

TEST_F(RangeFilterTest, PullExposurePreservesTheLocalRangeApproximation) {
  TermQuery term("body_w", "dense_a");
  NumericPredicateQuery range("proof_points_i", 0, N - 1);
  std::array<Query*, 1> required{&term}, filters{&range};
  BooleanQuery query(required, {}, {}, filters);
  MemPool pool;
  Query::Context context(pool, *reader);
  SkipStatsScope stats;
  auto* supplier = query.createWeight(context, 0)->scorerSupplier(pool, reader->segments()[0]);
  auto build = supplier->makePlanContext(Query::Demand{});
  build.expectedSpan = 64;
  auto* scorer = supplier->resolve(pool, build)->build(pool);
  EXPECT_EQ(1, SkipStats::numericPredicateZoneArms);
  EXPECT_EQ(0, SkipStats::numericPredicateSparseVerifyArms);
  std::array<uint64_t, 1> bits{};
  scorer->fillWindowBits(bits, 4096, 4160);
  EXPECT_EQ(0x0001000100010001ULL, bits[0]);
  EXPECT_EQ(1, SkipStats::numericPredicateClassifiedBlocks);
}
