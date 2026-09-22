#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

# This utility only normalizes the generated LaTeX fragments in this preview.
# Frozen Markdown chapters and evidence files are never edited by this script.

my %maps = (
    1 => {
        '1-3' => '\\citep{nouira2018,xie2022,jiao2023}',
        '1'   => '\\citep{nouira2018}',
        '2'   => '\\citep{xie2022}',
        '2-4,8' => '\\citep{xie2022,jiao2023,jiao2024,zeni2025}',
        '2,3,8' => '\\citep{xie2022,jiao2023,zeni2025}',
        '3'   => '\\citep{jiao2023}',
        '4'   => '\\citep{jiao2024}',
        '4,9' => '\\citep{jiao2024,okabe2025}',
        '5'   => '\\citep{yang2024}',
        '6'   => '\\citep{xiao2023}',
        '7'   => '\\citep{antunes2024}',
        '8'   => '\\citep{zeni2025}',
        '9'   => '\\citep{okabe2025}',
        '9,10' => '\\citep{okabe2025,ye2026}',
        '10'  => '\\citep{ye2026}',
        '11-13' => '\\citep{ho2020,song2021,ho2022}',
        '13'  => '\\citep{ho2022}',
        '14'  => '\\citep{merchant2023}',
        '14-19' => '\\citep{merchant2023,chen2022,batzner2022,batatia2022,deng2023,yang2024mattersim}',
        '15'  => '\\citep{chen2022}',
        '16'  => '\\citep{batzner2022}',
        '17'  => '\\citep{batatia2022}',
        '18'  => '\\citep{deng2023}',
        '19'  => '\\citep{yang2024mattersim}',
        '20'  => '\\citep{wang2024}',
        '21'  => '\\citep{long2024}',
    },
    2 => {
        '1' => '\\citep{zeni2025}',
        '2' => '\\citep{ho2020}',
        '3' => '\\citep{song2021}',
        '4' => '\\citep{ho2022}',
        '5' => '\\citep{gasteiger2021}',
        '6' => '\\citep{yang2024mattersim}',
        '7' => '\\citep{deng2023}',
    },
    3 => {
        '1' => '\\citep{zeni2025}',
        '2' => '\\citep{ho2022}',
        '3' => '\\citep{yang2024mattersim}',
        '4' => '\\citep{deng2023}',
    },
    4 => {
        '1' => '\\citep{zeni2025}',
        '2' => '\\citep{ho2020}',
        '3' => '\\citep{song2021}',
        '4' => '\\citep{yang2024mattersim}',
        '5' => '\\citep{deng2023}',
    },
);

for my $chapter (1 .. 6) {
    my $path = "contents/chap$chapter.tex";
    next unless -f $path;
    open my $in, '<:encoding(UTF-8)', $path or die "$path: $!";
    local $/;
    my $text = <$in>;
    close $in;

    # Convert the local numeric citation markers emitted by Pandoc to one
    # globally ordered natbib bibliography.  Other bracketed numeric values
    # (confidence intervals, arrays, etc.) are deliberately not touched.
    if (exists $maps{$chapter}) {
        for my $marker (sort { length($b) <=> length($a) } keys %{ $maps{$chapter} }) {
            my $replacement = $maps{$chapter}{$marker};
            # Pandoc escapes literal square brackets as ``{[}`` and
            # ``{]}``; match that exact representation rather than ordinary
            # LaTeX optional-argument brackets.
            $text =~ s/\{\[\}\Q$marker\E\{\]\}/$replacement/g;
        }
    }

    # The source chapters use a bold text line for table titles.  Convert the
    # generated longtables to the USTB template's bilingual caption form.
    # Captions belong inside longtable: captionof outside the environment
    # produces a detached title and the caption package warning
    # ``setcaptiontype outside box or environment``.
    $text =~ s/^\\textbf\{表(\d+)-(\d+)\s+([^{}]+)\}\s*$/\\captionof{table}{$3}\\label{tab:$1-$2}/mg;

    my %table_en = (
        '1-1' => 'Comparison of Representative Crystal Generation Methods',
        '1-2' => 'Research Questions and Technical Route of This Thesis',
        '2-1' => 'Main Symbols Used in This Thesis',
        '2-2' => 'Three Generation Fields of MatterGen',
        '2-3' => 'Roles of MatterSim and CHGNet in This Thesis',
        '2-4' => 'Main Evaluation Metrics Used in This Thesis',
        '3-1' => 'Summary of Online Adaptive CFG Exploration and Mechanistic Experiments',
        '3-2' => 'Fresh P0 Results of Stage-Calibrated CFG',
        '3-3' => 'Branch-Compatible Candidate Strategies',
        '3-4' => 'Test-Set Results of the Branch-Compatible Oracle',
        '3-5' => 'Final Confirmation Results for C1-128',
        '4-1' => 'Stage Reliability Diagnosis for Predicted Clean Structures at Late Timesteps',
        '4-2' => 'Frozen Configuration of RC-NFGD',
        '4-3' => 'Effect Reproduction on P0 and Formal32',
        '4-4' => 'Formal256 Main Results (relative changes and CIs use the favorable C0 minus RC-NFGD direction)',
        '4-5' => 'Independent CHGNet Proxy Evaluation on Formal256',
        '4-6' => 'Force-Direction Ablation (n=32)',
        '5-1' => 'Key Experiments, Sample Sizes, and Evidence Types',
        '5-2' => 'Final Confirmation Results for Innovation 1 on C1-128',
        '5-3' => 'RC-NFGD Formal256 Results',
        '5-4' => 'Cross-Proxy Results of MatterSim and CHGNet on Formal256',
        '5-5' => 'Key Ablations, Mechanistic Evidence, and Interpretation Boundaries',
        '5-6' => 'Computational Cost and Deployment Conditions',
        '5-7' => 'Final Claim--Evidence Mapping',
    );

    # First handle already-normalized captionof forms.  The generated
    # chapters wrap each longtable in a local group solely to suppress the
    # counter; that group is no longer needed once bicaption owns the counter.
    for my $id (keys %table_en) {
        my $en = $table_en{$id};
        $text =~ s{
            \\captionof\{table\}\{([^{}]*)\}\\label\{tab:\Q$id\E\}\n
            \{\\def\\LTcaptype\{none\}\s*%\s*do\s+not\s+increment\s+counter\s*\n
            (\\begin\{longtable\}\[\]\{.*?)(\n\\toprule)
        }{
            $2\\bicaption{$1}{$en}\\label{tab:$id}\\\\$3
        }gmsx;
    }
    $text =~ s/(\\end\{longtable\})\n\}\s*/$1\n/g;

    # Remove any stale local size switch left by an interrupted preview build.
    # Table sizing is kept uniform here; the official template's caption
    # typography is handled by the class and mixed table sizes look untidy.
    $text =~ s/\n(?:[ \t]*\\scriptsize\n)+(?=\\begin\{longtable\})/\n/g;
    $text =~ s/\\scriptsize\n(?=\\bicaption\{)/ /g;
    $text =~ s{(\@\{\}\})\\scriptsize}{$1}g;

    # A longtable repeats everything up to \endhead on continuation pages.
    # Put the bilingual caption and the first header in a first-head section,
    # then repeat only the three-line header.  This prevents duplicate .lot
    # entries and multiply-defined labels for tables that span pages.  The
    # block-level guard also makes the normalizer idempotent when the build
    # script is run repeatedly.
    $text =~ s{(\\begin\{longtable\}.*?\\end\{longtable\})}{
        my $block = $1;
        # Earlier preview builds could add more than one first-head section;
        # retain the first caption/header and the final continuation header.
        $block =~ s{(\\endfirsthead)(?s:.*)\\endfirsthead(.*?\\endhead)}{$1$2}s;
        if ($block !~ /\\endfirsthead/) {
            $block =~ s{
                (\\toprule\n\s*\\noalign\x7b\x7d\n)
                (.*?)
                (\\midrule\\noalign\x7b\x7d\n\\endhead)
            }{
                $1$2\\midrule\\noalign{}
\\endfirsthead
                $1$2$3
            }s;
        }
        $block;
    }gmsxe;

    my %figure_en = (
        '本文总体技术路线' => 'Overall Technical Route of This Thesis',
        '晶体结构的晶胞、分数坐标与周期映射示意图' => 'Schematic of Crystal Cells, Fractional Coordinates, and Periodic Mapping',
        '扩散模型前向加噪与反向生成示意图' => 'Forward Noising and Reverse Generation in Diffusion Models',
        '条件与无条件得分构成 CFG 的示意图' => 'Construction of CFG from Conditional and Unconditional Scores',
        'MatterGen 三字段条件生成总体流程' => 'Overall Workflow of MatterGen Three-Field Conditional Generation',
        '神经势的结构---能量---原子力关系示意图' => 'Relationship among Structure, Energy, and Atomic Forces in a Neural Potential',
        '创新点1的证据演化：方框表示各阶段冻结结论，而非单调性能曲线' => 'Evidence Evolution of Innovation 1: Boxes Denote Frozen Stage Conclusions Rather Than a Monotonic Performance Curve',
        'C1-128 四种方法的 Property MAE（release ID: I1-F1）' => 'Property MAE of Four Methods on C1-128 (release ID: I1-F1)',
        'C1-128 中 C0、Fixed-K2、Linear-K2 与 Random-K2 的 Property MAE' => 'Property MAE of C0, Fixed-K2, Linear-K2, and Random-K2 on C1-128',
        '创新点1从在线自适应探索到最终确认的证据演进（release ID: I1-F6）' => 'Evidence Evolution of Innovation 1 from Online Adaptive Exploration to Final Confirmation (release ID: I1-F6)',
        'C0、Fixed-K2、Linear-K2 与 Random-K2 的 Property MAE' => 'Property MAE of C0, Fixed-K2, Linear-K2, and Random-K2',
        'Fixed-K2 与 C0 的逐样本属性误差' => 'Per-Sample Property Error of Fixed-K2 and C0',
        'Fixed-K2 相对 C0 的 20 000 次配对自助法增益分布' => '20,000 Paired-Bootstrap Gain Distribution of Fixed-K2 Relative to C0',
        'C1-128 的代理质量护栏结果' => 'Proxy Quality Guardrails on C1-128',
        'C0、Fixed-K2 及 Phase B 数据构建的计算量比较' => 'Computational Cost Comparison of C0, Fixed-K2, and Phase B Data Construction',
        'RC-NFGD 在线引导流程（release ID: I2-F1）' => 'Online Guidance Workflow of RC-NFGD (release ID: I2-F1)',
        'Formal256 主要代理结构指标（release ID: I2-F2）' => 'Main Proxy Structural Metrics on Formal256 (release ID: I2-F2)',
        'Formal256 配对改善分布（release ID: I2-F3）' => 'Paired Improvement Distribution on Formal256 (release ID: I2-F3)',
        'P0、Formal32 与 Formal256 效应方向一致性（release ID: I2-F4）' => 'Consistency of Effect Direction across P0, Formal32, and Formal256 (release ID: I2-F4)',
        'P0、Formal32 与 Formal256 的 MaxF 效应一致性（release ID: I2-F4）' => 'Consistency of MaxF Effects across P0, Formal32, and Formal256 (release ID: I2-F4)',
        'RC-NFGD 在 Formal256 上的质量护栏（release ID: I2-F6）' => 'Quality Guardrails of RC-NFGD on Formal256 (release ID: I2-F6)',
        'MatterSim 与 CHGNet 的逐样本改善比较（release ID: I2-F5）' => 'Per-Sample Improvement Comparison between MatterSim and CHGNet (release ID: I2-F5)',
        'C0、Fixed-K2 与 Phase B 数据构建的计算量比较（release ID: I1-F5）' => 'Computational Cost Comparison of C0, Fixed-K2, and Phase B Data Construction (release ID: I1-F5)',
    );
    for my $cn (keys %figure_en) {
        my $en = $figure_en{$cn};
        $text =~ s{\\caption\{\Q$cn\E\}}{\\bicaption{$cn}{$en}}g;
    }

    # Keep long technical tokens breakable in narrow table columns.  These
    # are TeX line-breaking hints only; the rendered scientific text is
    # unchanged.
    $text =~ s/(?:\\allowbreak)+/\\allowbreak/g;
    $text =~ s/MatterSim/Matter\\allowbreak Sim/g;
    $text =~ s/Spearman=(?!\\allowbreak)/Spearman=\\allowbreak/g;
    $text =~ s/0\.028799→0\.020041/0.028799→\\allowbreak 0.020041/g;
    $text =~ s/MECHANISM/MECHAN\\allowbreak ISM/g;
    $text =~ s/Counterfactual/Counter\\allowbreak factual/g;
    $text =~ s/SUPPORTED/SUP\\allowbreak PORTED/g;
    $text =~ s/CONFIRMED/CONFIR\\allowbreak MED/g;
    $text =~ s/Random-K2/Random-\\allowbreak K2/g;
    $text =~ s/Formal32/Formal\\allowbreak 32/g;
    $text =~ s/V1\/V2\/V4\/V5(?!\\allowbreak)/V1\/V2\/V4\/V5\\allowbreak/g;

    open my $out, '>:encoding(UTF-8)', $path or die "$path: $!";
    print {$out} $text;
    close $out;
}
