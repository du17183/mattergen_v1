#!/usr/bin/env perl
use strict;
use warnings;
use utf8;

# Presentation-only post-processing for the polished preview.  The source
# chapters and frozen release artifacts are not edited.  Running this after
# normalize_content.pl makes the preview reproducible and idempotent.

for my $chapter (1 .. 6) {
    my $path = "contents/chap$chapter.tex";
    next unless -f $path;
    open my $in, '<:encoding(UTF-8)', $path or die "$path: $!";
    my @lines = <$in>;
    close $in;

    my $figure_index = 0;
    my $in_figure = 0;
    my @out;
    for my $line (@lines) {
        # The conceptual diagrams in Chapters 1--2 have vector masters in
        # thesis_release/concept_figures_polished.  Keep the original PNG/SVG
        # assets untouched, but use the PDF masters in this preview so print
        # scaling does not introduce raster edges.
        $line =~ s{../../../thesis/chapter1/figures/figure1_1_technical_route\.png}{../../../thesis_release/concept_figures_polished/figure1_1_technical_route.pdf}g;
        $line =~ s{../../../thesis/chapter2/figures/fig2_1_crystal_representation\.png}{../../../thesis_release/concept_figures_polished/fig2_1_crystal_representation.pdf}g;
        $line =~ s{../../../thesis/chapter2/figures/fig2_2_diffusion_process\.png}{../../../thesis_release/concept_figures_polished/fig2_2_diffusion_process.pdf}g;
        $line =~ s{../../../thesis/chapter2/figures/fig2_3_cfg\.png}{../../../thesis_release/concept_figures_polished/fig2_3_cfg.pdf}g;
        $line =~ s{../../../thesis/chapter2/figures/fig2_4_mattergen_overview\.png}{../../../thesis_release/concept_figures_polished/fig2_4_mattergen_overview.pdf}g;
        $line =~ s{../../../thesis/chapter2/figures/fig2_5_mlip_energy_force\.png}{../../../thesis_release/concept_figures_polished/fig2_5_mlip_energy_force.pdf}g;
        if ($line =~ /\\begin\{figure\}(?!\[)/) {
            $figure_index++;
            $in_figure = 1;
        }
        if ($in_figure) {
            # Release IDs remain in the release manifest and file name, but
            # should not pollute the visible caption, accessibility text, or
            # list of figures.
            $line =~ s/\s*[（(]release ID:\s*I[12]-F\d+[）)]//g;
            $line =~ s/\s*release ID:\s*I[12]-F\d+//g;
            if ($line =~ /\\bicaption\{/) {
                my $label = "\\label{fig:$chapter-$figure_index}";
                $line =~ s/(\\bicaption\{.*\})\s*$/$1$label\n/ unless $line =~ /\\label\{fig:/;
            }
        }
        if ($in_figure && $line =~ /\\end\{figure\}/) {
            $in_figure = 0;
        }
        push @out, $line;
    }

    # Add compact, bilingual group headers to the widest result tables.  The
    # group row is a presentation aid only: it does not rename, reorder, or
    # otherwise alter any data cell.  A marker makes this transformation
    # idempotent when the preview is rebuilt.
    my %groups = (
        '4-4' => [
            '\\multicolumn{1}{l}{\\makecell[l]{\\textbf{指标}\\\\[-1pt]\\scriptsize Metric}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{队列均值}\\\\[-1pt]\\scriptsize Means}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{效应与区间}\\\\[-1pt]\\scriptsize Effect + CI}} & '
                . '\\multicolumn{1}{c}{\\makecell{\\textbf{配对方向}\\\\[-1pt]\\scriptsize Paired}} \\\\',
            '\\cmidrule(lr){1-1}\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}',
        ],
        '5-2' => [
            '\\multicolumn{1}{l}{\\makecell[l]{\\textbf{方法}\\\\[-1pt]\\scriptsize Method}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{目标与能量}\\\\[-1pt]\\scriptsize Target}} & '
                . '\\multicolumn{5}{c}{\\makecell{\\textbf{质量护栏}\\\\[-1pt]\\scriptsize Guardrails}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{回退与成本}\\\\[-1pt]\\scriptsize Fallback + cost}} \\\\',
            '\\cmidrule(lr){1-1}\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\\cmidrule(lr){9-10}',
        ],
        '5-3' => [
            '\\multicolumn{1}{l}{\\makecell[l]{\\textbf{指标}\\\\[-1pt]\\scriptsize Metric}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{队列均值}\\\\[-1pt]\\scriptsize Means}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{效应与区间}\\\\[-1pt]\\scriptsize Effect + CI}} & '
                . '\\multicolumn{1}{c}{\\makecell{\\textbf{配对方向}\\\\[-1pt]\\scriptsize Paired}} & '
                . '\\multicolumn{1}{l}{\\makecell[l]{\\textbf{解释}\\\\[-1pt]\\scriptsize Interpretation}} \\\\',
            '\\cmidrule(lr){1-1}\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-7}',
        ],
        '5-4' => [
            '\\multicolumn{2}{l}{\\makecell[l]{\\textbf{评价对象}\\\\[-1pt]\\scriptsize Evaluator + metric}} & '
                . '\\multicolumn{2}{c}{\\makecell{\\textbf{队列均值}\\\\[-1pt]\\scriptsize Means}} & '
                . '\\multicolumn{3}{c}{\\makecell{\\textbf{效应与区间}\\\\[-1pt]\\scriptsize Effect + CI}} & '
                . '\\multicolumn{1}{l}{\\makecell[l]{\\textbf{证据}\\\\[-1pt]\\scriptsize Evidence}} \\\\',
            '\\cmidrule(lr){1-2}\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}',
        ],
    );
    if (exists $groups{"$chapter-4"} || exists $groups{"$chapter-2"} || exists $groups{"$chapter-3"}) {
        my $text = join('', @out);
        for my $id (keys %groups) {
            next unless $id =~ /^$chapter-/;
            my ($group_row, $rules) = @{ $groups{$id} };
            # Remove a prior generated group row/rule before inserting it.
            $text =~ s/^% POLISHED-GROUP-HEADER-$id\n.*?^% END-POLISHED-GROUP-HEADER-$id\n//msg;
            # Insert immediately after the first ``\noalign{}`` following the
            # table's top rule.  Using string offsets here avoids relying on
            # Perl's replacement captures (and keeps the pass idempotent).
            my $caption_pos = index($text, "\\label{tab:$id}");
            next if $caption_pos < 0;
            my $top_pos = index($text, "\\toprule", $caption_pos);
            next if $top_pos < 0;
            my $anchor = "\\noalign{}\n";
            my $anchor_pos = index($text, $anchor, $top_pos);
            next if $anchor_pos < 0;
            my $insert_at = $anchor_pos + length($anchor);
            my $insert = "% POLISHED-GROUP-HEADER-$id\n$group_row\n$rules\\\\\n% END-POLISHED-GROUP-HEADER-$id\n";
            substr($text, $insert_at, 0) = $insert;

            # Repeat the same row on continuation pages.  The second top rule
            # occurs after \endfirsthead; it is the only header that matters
            # for a longtable split across pages.
            my $first_end = index($text, "\\endfirsthead", $insert_at);
            if ($first_end >= 0) {
                my $cont_top = index($text, "\\toprule", $first_end);
                my $cont_anchor = index($text, $anchor, $cont_top);
                if ($cont_top >= 0 && $cont_anchor >= 0) {
                    my $cont_at = $cont_anchor + length($anchor);
                    substr($text, $cont_at, 0) = "% POLISHED-GROUP-HEADER-$id\n$group_row\n$rules\\\\\n% END-POLISHED-GROUP-HEADER-$id\n";
                }
            }
        }

        # Table 5-2 is the only ten-column result table whose body still has
        # a few unbreakable tokens at the class's deliberately zero inter-
        # column spacing.  Keep the table data and precision unchanged, and
        # use explicit first-column line breaks for the two method names.  The
        # change is local to this preview and does not alter the thesis-wide
        # font size or the meaning of a value.
        if ($chapter == 5) {
            # Undo only presentation wrappers from a previous pass before
            # applying them to the intended table block.  This prevents a
            # method name in Table 5-1/5-6 from being changed accidentally.
            $text =~ s/\\makecell\[l\]\{Linear-\\\\K2\}/Linear-K2/g;
            $text =~ s/\\makecell\[l\]\{Random-\\\\K2\}/Random-\\allowbreak K2/g;
            $text =~ s/% POLISHED-TABLE-FONT-5-2\n\\scriptsize\n//g;
            my $tab52_caption = "\\label{tab:5-2}\\\\\n";
            my $p = index($text, $tab52_caption);
            my $q = $p >= 0 ? index($text, "\\end{longtable}", $p) : -1;
            if ($p >= 0 && $q > $p) {
                my $block = substr($text, $p, $q - $p);
                $block =~ s/Linear-K2 &/\\makecell[l]{Linear\\\\-K2} &/g;
                $block =~ s/Random-\\allowbreak K2 &/\\makecell[l]{Random\\\\-K2} &/g;
                substr($text, $p, $q - $p) = $block;
            }
        }
        @out = split(/(?<=\n)/, $text);
    }

    open my $out, '>:encoding(UTF-8)', $path or die "$path: $!";
    print {$out} @out;
    close $out;
}

print "POLISHED_LATEX_FRAGMENT_NORMALIZATION=PASS\n";
